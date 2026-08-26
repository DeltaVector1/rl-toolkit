import argparse
import gzip
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


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
        self.range_gzip = bool(self.headers.get("Range")) and "gzip" in accepted
        self.send_header("Content-Type", self.guess_type(path))
        if self.range_gzip:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("X-Prime-Raw-Length", str(end - start + 1))
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
        if self.range_gzip:
            with gzip.GzipFile(
                fileobj=output, mode="wb", compresslevel=1, mtime=0
            ) as compressed:
                self._copy_range(source, compressed)
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
