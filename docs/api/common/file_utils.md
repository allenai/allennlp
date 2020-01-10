# allennlp.common.file_utils

Utilities for working with the local dataset cache.

## url_to_filename
```python
url_to_filename(url:str, etag:str=None) -> str
```

Convert `url` into a hashed filename in a repeatable way.
If `etag` is specified, append its hash to the url's, delimited
by a period.

## filename_to_url
```python
filename_to_url(filename:str, cache_dir:str=None) -> Tuple[str, str]
```

Return the url and etag (which may be ``None``) stored for `filename`.
Raise ``FileNotFoundError`` if `filename` or its stored metadata do not exist.

## cached_path
```python
cached_path(url_or_filename:Union[str, pathlib.Path], cache_dir:str=None) -> str
```

Given something that might be a URL (or might be a local path),
determine which. If it's a URL, download the file and cache it, and
return the path to the cached file. If it's already a local path,
make sure the file exists and then return the path.

## is_url_or_existing_file
```python
is_url_or_existing_file(url_or_filename:Union[str, pathlib.Path, NoneType]) -> bool
```

Given something that might be a URL (or might be a local path),
determine check if it's url or an existing file path.

## split_s3_path
```python
split_s3_path(url:str) -> Tuple[str, str]
```
Split a full s3 path into the bucket name and path.
## s3_request
```python
s3_request(func:Callable)
```

Wrapper function for s3 requests in order to create more helpful error
messages.

## s3_etag
```python
s3_etag(url:str) -> Union[str, NoneType]
```
Check ETag on S3 object.
## s3_get
```python
s3_get(url:str, temp_file:IO) -> None
```
Pull a file directly from S3.
## session_with_backoff
```python
session_with_backoff() -> requests.sessions.Session
```

We ran into an issue where http requests to s3 were timing out,
possibly because we were making too many requests too quickly.
This helper function returns a requests session that has retry-with-backoff
built in.
see stackoverflow.com/questions/23267409/how-to-implement-retry-mechanism-into-python-requests-library

## get_from_cache
```python
get_from_cache(url:str, cache_dir:str=None) -> str
```

Given a URL, look for the corresponding dataset in the local cache.
If it's not there, download it. Then return the path to the cached file.

## read_set_from_file
```python
read_set_from_file(filename:str) -> Set[str]
```

Extract a de-duped collection (set) of text from a file.
Expected file format is one item per line.

