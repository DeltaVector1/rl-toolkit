import argparse
import ctypes
import ctypes.util
import gzip
import os
import struct
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


_ZSTD_LEVEL = 1
_ZSTD_FRAME_BYTES = 8 * 1024 * 1024
_zstd = None


def _zstd_library():
    global _zstd
    if _zstd is None:
        library = ctypes.util.find_library("zstd") or "libzstd.so.1"
        try:
            loaded = ctypes.CDLL(library)
        except OSError as error:
            raise OSError("zstd adapter transfer requires libzstd") from error
        size_t = ctypes.c_size_t
        loaded.ZSTD_compressBound.argtypes = [size_t]
        loaded.ZSTD_compressBound.restype = size_t
        loaded.ZSTD_compress.argtypes = [
            ctypes.c_void_p,
            size_t,
            ctypes.c_void_p,
            size_t,
            ctypes.c_int,
        ]
        loaded.ZSTD_compress.restype = size_t
        loaded.ZSTD_isError.argtypes = [size_t]
        loaded.ZSTD_isError.restype = ctypes.c_uint
        loaded.ZSTD_getErrorName.argtypes = [size_t]
        loaded.ZSTD_getErrorName.restype = ctypes.c_char_p
        _zstd = loaded
    return _zstd


def _zstd_compress(raw: bytes) -> bytes:
    library = _zstd_library()
    capacity = int(library.ZSTD_compressBound(len(raw)))
    output = ctypes.create_string_buffer(capacity)
    result = int(library.ZSTD_compress(output, capacity, raw, len(raw), _ZSTD_LEVEL))
    if library.ZSTD_isError(result):
        message = library.ZSTD_getErrorName(result).decode("utf-8", errors="replace")
        raise OSError(f"zstd adapter compression failed: {message}")
    return output.raw[:result]


class GzipRangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        try:
            source = open(path, "rb")
        except OSError:
            self.send_error(404)
            return None
        size = os.fstat(source.fileno()).st_size
        start, end = 0, size - 1
        if value := self.headers.get("Range"):
            try:
                start_text, end_text = value.removeprefix("bytes=").split("-", 1)
                start = int(start_text)
                end = min(int(end_text), end) if end_text else end
                if start > end:
                    raise ValueError
            except ValueError:
                source.close()
                self.send_error(416)
                return None
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        accepted = {
            value.split(";", 1)[0].strip().lower()
            for value in self.headers.get("Accept-Encoding", "").split(",")
        }
        self.range_codec = None
        if self.headers.get("Range"):
            if "zstd" in accepted:
                self.range_codec = "zstd"
            elif "gzip" in accepted:
                self.range_codec = "gzip"
        self.send_header("Content-Type", self.guess_type(path))
        if self.range_codec:
            self.send_header("Content-Encoding", self.range_codec)
            self.send_header("X-Prime-Raw-Length", str(end - start + 1))
            if self.range_codec == "zstd":
                self.send_header("X-Prime-Codec", "zstd-frames-v1")
            self.send_header("Vary", "Accept-Encoding")
        else:
            self.send_header("Content-Length", str(end - start + 1))
        self.send_header(
            "Last-Modified",
            self.date_time_string(os.fstat(source.fileno()).st_mtime),
        )
        self.end_headers()
        source.seek(start)
        self.remaining = end - start + 1
        return source

    def copyfile(self, source, output):
        if not hasattr(self, "remaining"):
            return super().copyfile(source, output)
        if self.range_codec == "gzip":
            with gzip.GzipFile(
                fileobj=output, mode="wb", compresslevel=1, mtime=0
            ) as compressed:
                self._copy_range(source, compressed)
            return
        if self.range_codec == "zstd":
            remaining = self.remaining
            while remaining:
                raw = source.read(min(remaining, _ZSTD_FRAME_BYTES))
                if not raw:
                    break
                encoded = _zstd_compress(raw)
                output.write(struct.pack("!II", len(encoded), len(raw)))
                output.write(encoded)
                remaining -= len(raw)
            self.remaining = remaining
            return
        self._copy_range(source, output)

    def _copy_range(self, source, output):
        while self.remaining:
            chunk = source.read(min(self.remaining, 16 * 1024 * 1024))
            if not chunk:
                break
            output.write(chunk)
            self.remaining -= len(chunk)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--port", type=int, default=8101)
    args = parser.parse_args()
    ThreadingHTTPServer(
        ("127.0.0.1", args.port),
        partial(GzipRangeHandler, directory=args.directory),
    ).serve_forever()


if __name__ == "__main__":
    main()
