"""Fail closed before local tests can reach an operational boundary."""

import ctypes
import errno
import os
from pathlib import Path
import sys
from unittest import mock

import training_ops

installed = False


def install():
    global installed
    if installed:
        return
    os.environ.pop("BASH_ENV", None)
    os.environ.pop("ENV", None)
    # The kernel filter is inherited by every shell and child interpreter.
    # Do not run this suite without it: Python mocks do not constrain children.
    library = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    context = library.seccomp_init(0x7fff0000)
    if not context:
        raise RuntimeError("cannot create local test network filter")
    try:
        for name in (b"connect", b"sendto", b"sendmsg", b"sendmmsg"):
            number = library.seccomp_syscall_resolve_name(name)
            if number < 0 or library.seccomp_rule_add(context, 0x00050000 | errno.EPERM, number, 0) != 0:
                raise RuntimeError("cannot install local test network rule")
        if library.seccomp_load(context) != 0:
            raise RuntimeError("cannot activate local test network filter")
    finally:
        library.seccomp_release(context)

    original_process = training_ops._run_process
    original_ssh = training_ops._ssh
    def process(*args, **kwargs):
        if not isinstance(training_ops.subprocess.Popen, mock.Mock):
            raise AssertionError("unmocked training_ops process execution is forbidden")
        return original_process(*args, **kwargs)
    def ssh(*args, **kwargs):
        if not isinstance(training_ops._run_process, mock.Mock) and not isinstance(training_ops.subprocess.Popen, mock.Mock):
            raise AssertionError("unmocked training_ops SSH is forbidden")
        return original_ssh(*args, **kwargs)
    def deploy(*args, **kwargs):
        raise AssertionError("unmocked deployment is forbidden")
    training_ops._run_process = process
    training_ops._ssh = ssh
    training_ops.deploy = deploy

    def audit(event, arguments):
        if event == "subprocess.Popen":
            command = arguments[1]
            executable = command[0] if isinstance(command, (list, tuple)) else command
            if any(Path(str(value)).name in {"ssh", "scp", "rsync", "curl", "wget"} for value in (arguments[0], executable)):
                raise AssertionError("live-capable child command is forbidden in local tests")
    sys.addaudithook(audit)
    installed = True
