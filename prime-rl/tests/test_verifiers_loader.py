import contextvars
import unittest


class VerifiersLoaderImportTest(unittest.TestCase):
    def test_skip_plugin_install_comes_from_v1_loader(self):
        from verifiers.v1.loaders import skip_plugin_install

        self.assertIsInstance(skip_plugin_install, contextvars.ContextVar)
        token = skip_plugin_install.set(True)
        try:
            self.assertTrue(skip_plugin_install.get())
        finally:
            skip_plugin_install.reset(token)


if __name__ == "__main__":
    unittest.main()
