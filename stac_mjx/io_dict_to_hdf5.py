# copied from https://codereview.stackexchange.com/a/121308 (and slightly modified for updated h5py, Elliott Abe)
import numpy as np
import h5py
import jax.numpy as jnp
#import os

def save(filename, dic, compression='gzip', compression_opts=5, chunked_datasets=None):
    """
    saves a python dictionary or list, with items that are themselves either
    dictionaries or lists or (in the case of tree-leaves) numpy arrays
    or basic scalar types (int/float/str/bytes) in a recursive
    manner to an hdf5 file, with an intact hierarchy.

    Parameters:
    -----------
    filename : str
        Path to the HDF5 file to save
    dic : dict/list
        Data structure to save
    compression : str, optional
        Compression algorithm ('gzip', 'lzf', 'szip', None). Default is 'gzip'
    compression_opts : int, optional
        Compression level (0-9 for gzip). Default is 5
    chunked_datasets : dict, optional
        Dictionary specifying chunk shapes for specific datasets.
        Format: {'dataset_name': chunk_shape, ...}
        Example: {'qpos': (1, None, None)} means chunk by first dimension only
    """
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic, compression,
                                               compression_opts=compression_opts,
                                               chunked_datasets=chunked_datasets or {})

def recursively_save_dict_contents_to_group(h5file, path, dic, compression='gzip', compression_opts=5, chunked_datasets=None):
    """
    Recursively save dictionary contents to HDF5 group with compression support.

    Parameters:
    -----------
    h5file : h5py.File
        The HDF5 file object
    path : str
        Current path in the HDF5 hierarchy
    dic : dict/list/object
        Data structure to save
    compression : str, optional
        Compression algorithm ('gzip', 'lzf', 'szip', None). Default is 'gzip'
    compression_opts : int, optional
        Compression level (0-9 for gzip). Default is 5
    chunked_datasets : dict, optional
        Dictionary specifying chunk shapes for specific datasets
    """
    if chunked_datasets is None:
        chunked_datasets = {}
    if isinstance(dic,dict):
        iterator = dic.items()
    elif isinstance(dic,list):
        iterator = enumerate(dic)
    elif isinstance(dic,object):
        iterator = dic.__dict__.items()
    else:
        ValueError('Cannot save %s type' % type(dic))

    for key, item in iterator: #dic.items():
        if isinstance(dic,(list, tuple)):
            key = str(key)
        if isinstance(item, (jnp.ndarray, np.ndarray, np.int64, np.float64, int, float, str, bytes)):
            # Use create_dataset with compression for arrays and numeric data
            if isinstance(item, (jnp.ndarray, np.ndarray)) and item.size > 1:
                # Check if this dataset should use chunking
                dataset_name = key
                if dataset_name in chunked_datasets:
                    chunk_shape = chunked_datasets[dataset_name]
                    # Replace None values with actual dimensions
                    if chunk_shape is not None:
                        chunks = tuple(
                            item.shape[i] if chunk_shape[i] is None else chunk_shape[i]
                            for i in range(len(chunk_shape))
                        )
                        h5file.create_dataset(path + key, data=item,
                                           compression=compression, compression_opts=compression_opts,
                                           chunks=chunks)
                    else:
                        # chunked_datasets[key] = None means no chunking
                        h5file.create_dataset(path + key, data=item,
                                           compression=compression, compression_opts=compression_opts)
                else:
                    # Default: apply compression for arrays with more than one element
                    h5file.create_dataset(path + key, data=item,
                                       compression=compression, compression_opts=compression_opts)
            else:
                # For scalars and single-element arrays, compression may not be beneficial
                h5file[path + key] = item
        elif isinstance(item, (dict,list,object)): # or isinstance(item,list) or isinstance(item,tuple):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item, compression,
                                                 compression_opts=compression_opts,
                                                 chunked_datasets=chunked_datasets)
        else:
            raise ValueError('Cannot save %s type'%type(item))

def convert_dict_to_list_if_appropriate(d):
    """Convert a dictionary with string integer keys back to a list if appropriate"""
    if not isinstance(d, dict):
        return d
    
    # Check if all keys can be converted to integers
    try:
        keys_as_ints = [int(k) for k in d.keys()]
        if sorted(keys_as_ints) == list(range(len(keys_as_ints))):
            # Keys are consecutive integers starting from 0, convert to list
            result = [None] * len(d)
            for k, v in d.items():
                result[int(k)] = v
            return result
    except (ValueError, TypeError):
        pass
    
    return d

def recursively_convert_appropriate_dicts_to_lists(data):
    """Recursively convert dictionaries that should be lists back to lists"""
    if isinstance(data, dict):
        # First, recursively process all values
        processed_data = {k: recursively_convert_appropriate_dicts_to_lists(v) for k, v in data.items()}
        # Then check if this dictionary should be converted to a list
        return convert_dict_to_list_if_appropriate(processed_data)
    elif isinstance(data, list):
        return [recursively_convert_appropriate_dicts_to_lists(item) for item in data]
    else:
        return data

def load(filename, ASLIST=False, enable_jax=False, auto_convert_lists=True):
    """
    Default: load a hdf5 file (saved with io_dict_to_hdf5.save function above) as a hierarchical
    python dictionary (as described in the doc_string of io_dict_to_hdf5.save).
    
    Parameters:
    - ASLIST: if True, loads the top level as a list (requires integer convertible keys)
    - enable_jax: if True, converts numeric data to JAX arrays while preserving strings
    - auto_convert_lists: if True, automatically detects and converts dictionaries that 
      were originally lists back to lists (based on consecutive integer string keys)
    
    Both ASLIST and enable_jax can be used together - JAX conversion will be applied to appropriate data types
    while maintaining list structure for containers.
    """
    with h5py.File(filename, 'r') as h5file:
        out = recursively_load_dict_contents_from_group(h5file, '/', enable_jax=enable_jax)
        
        # Apply automatic list conversion if requested
        if auto_convert_lists:
            out = recursively_convert_appropriate_dicts_to_lists(out)
        
        # Apply top-level ASLIST conversion if requested
        if ASLIST:
            outl = [None for l in range(len(out.keys()))]
            for key, item in out.items():
                outl[int(key)] = item
            out = outl
            
        return out


def save_reference_clips_chunked(filename, reference_clips_dict, compression='gzip', compression_opts=3,
                                clips_per_chunk=1):
    """
    Save ReferenceClips data with optimal chunking for parallel single-clip access.

    This function automatically configures chunking for ReferenceClips data arrays:
    - qpos, qvel, xpos, xquat are chunked by clip for efficient single-clip loading
    - Other data (clip_lengths, qpos_names) are stored without chunking

    Parameters:
    -----------
    filename : str
        Path to the HDF5 file to save
    reference_clips_dict : dict
        Dictionary from ReferenceClips.to_dict()
    compression : str, optional
        Compression algorithm. Default is 'gzip'
    compression_opts : int, optional
        Compression level (0-9). Default is 5
    clips_per_chunk : int, optional
        Number of clips per chunk (1 = optimal for single-clip access). Default is 1

    Example:
    --------
    # Save with optimal chunking for single-clip access
    clips = ReferenceClips.from_path("data.h5")
    save_reference_clips_chunked("data_chunked.h5", clips.to_dict())

    # Load single clip efficiently
    hdf5_clips = HDF5ReferenceClips("data_chunked.h5")
    single_clip = hdf5_clips.load_single_clip(0)  # Fast - reads only 1 chunk
    """

    # Define optimal chunking strategy for ReferenceClips
    chunked_datasets = {}

    # Check shapes to configure chunking
    if 'qpos' in reference_clips_dict:
        qpos_shape = reference_clips_dict['qpos'].shape
        # Chunk by clip: (clips_per_chunk, all_frames, all_features)
        chunked_datasets['qpos'] = (clips_per_chunk, None, None)

    if 'qvel' in reference_clips_dict:
        chunked_datasets['qvel'] = (clips_per_chunk, None, None)

    if 'xpos' in reference_clips_dict:
        chunked_datasets['xpos'] = (clips_per_chunk, None, None, None)

    if 'xquat' in reference_clips_dict:
        chunked_datasets['xquat'] = (clips_per_chunk, None, None, None)

    # clip_lengths and qpos_names don't need chunking (small 1D arrays)

    print(f"Saving ReferenceClips with chunking strategy:")
    for dataset, chunk_shape in chunked_datasets.items():
        if dataset in reference_clips_dict:
            data_shape = reference_clips_dict[dataset].shape
            actual_chunks = tuple(
                data_shape[i] if chunk_shape[i] is None else chunk_shape[i]
                for i in range(len(chunk_shape))
            )
            print(f"  {dataset}: {data_shape} -> chunks={actual_chunks}")

    # Save with chunking
    save(filename, reference_clips_dict, compression=compression,
         compression_opts=compression_opts, chunked_datasets=chunked_datasets)

    print(f"Saved to {filename} with optimal chunking for parallel single-clip access")


def recursively_load_dict_contents_from_group(h5file, path, enable_jax=False):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            data = item[()]
            
            # Handle string/bytes conversion
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            elif isinstance(data, np.ndarray) and data.dtype.kind in ['S', 'U']:
                # Handle numpy string arrays
                if data.dtype.kind == 'S':  # byte strings
                    data = np.array([s.decode('utf-8') if isinstance(s, bytes) else s for s in data.flat]).reshape(data.shape)
                # Unicode strings (dtype.kind == 'U') are already handled correctly
            
            # Only convert to JAX if enable_jax is True AND it's not string data
            if enable_jax and isinstance(data, np.ndarray) and data.dtype.kind not in ['S', 'U']:
                ans[key] = jnp.asarray(data)
            elif enable_jax and isinstance(data, (int, float)) and not isinstance(data, (str, bytes)):
                ans[key] = jnp.asarray(data)
            else:
                ans[key] = data
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/', enable_jax=enable_jax)
    return ans
