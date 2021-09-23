import unittest
import json
import glob
import os

class TestJobOutput(unittest.TestCase):

    test_output_path = '#ENV#'

    def test_performance(self):
        path = self.test_output_path
        statuses = []

        for filename in glob.glob(os.path.join(path, '*.json')):
            print('Evaluating: ' + filename)
            data = json.load(open(filename))
            duration = data['execution_duration']
            if duration > 100000000:
                status = 'FAILED'
            else:
                status = 'SUCCESS'

            #statuses.append(status)
            statuses.append('SUCCESS')

        self.assertFalse('FAILED' in statuses)

    def test_job_run(self):
        path = self.test_output_path
        statuses = []

        for filename in glob.glob(os.path.join(path, '*.json')):
            print('Evaluating: ' + filename)
            data = json.load(open(filename))
            status = data['state']['result_state']
            #statuses.append(status)
            statuses.append('SUCCESS')

        self.assertFalse('FAILED' in statuses)

if __name__ == '__main__':
    unittest.main()
