import unittest
from lib.log import LoggerAdaptor, configure_logging
import logging
_logger = logging.getLogger("TestBase")

from config import settings, Settings

class UintTestConfig(Settings):
    pass

unittest_config = UintTestConfig("py_mapping.lib.settings")

# please see https://stackoverflow.com/questions/8518043/turn-some-print-off-in-python-unittest
class MyTestResult(unittest.TextTestResult):

    def addSuccess(self, test):
        unittest.TestResult.addSuccess(self, test)
        self.stream.write("Successful with <%s.%s>!\n\n\r" % (test.__class__.__name__, test._testMethodName))

    def addError(self, test, err):
        unittest.TestResult.addError(self, test, err)
        self.stream.write("An Error Found!\n\n\r")

    def addFailure(self, test, err):
        unittest.TestResult.addFailure(self, test, err)
        self.stream.write("An Failure Found!\n\n\r")


# see http://python.net/crew/tbryan/UnitTestTalk/slide30.html
class MyTestRunner(unittest.TextTestRunner):
    def _makeResult(self):
        return MyTestResult(self.stream, self.descriptions, self.verbosity)

def runTest(*test_cases):
    suite = unittest.TestSuite()
    for test_case in test_cases:
        assert isinstance(test_case, unittest.TestCase)
        suite.addTest(test_case)
    runner = MyTestRunner()
    runner.run(suite)
    return runner

def Program(filter_test_cases=None):
    # logging.basicConfig(level=logging.INFO, format=u"%(asctime)s [%(levelname)s]:%(filename)s, %(name)s, in line %(lineno)s >> %(message)s".encode('utf-8'))
    configure_logging(unittest_config.LOGGING)

    if unittest_config.ALL_TEST:
        unittest.main(testRunner=MyTestRunner)
    else:
        assert(filter_test_cases != None and hasattr(filter_test_cases, "__call__"))
        test_cases = filter_test_cases()

        # run tests
        runTest(test_cases)
    pass

if __name__ == "__main__":
    Program()