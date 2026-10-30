import subprocess
import unittest
from unittest.mock import patch

from scripts.hardware_report import gpu_info


class HardwareReportTests(unittest.TestCase):
    def test_large_driver_xml_and_nested_pcie_fields_are_preserved(self):
        xml = ('<nvidia_smi_log><driver_version>595.99.02</driver_version>'
               '<unused>' + 'x' * 45000 + '</unused><gpu><product_name>GPU</product_name>'
               '<pci><pci_bus_id>00000000:81:00.0</pci_bus_id><pci_gpu_link_info>'
               '<pcie_gen><max_link_gen>4</max_link_gen><current_link_gen>1</current_link_gen>'
               '</pcie_gen><link_widths><max_link_width>16x</max_link_width>'
               '<current_link_width>16x</current_link_width></link_widths>'
               '</pci_gpu_link_info></pci></gpu></nvidia_smi_log>')
        with patch('scripts.hardware_report.subprocess.run',
                   return_value=subprocess.CompletedProcess([], 0, xml, '')):
            result = gpu_info()
        self.assertEqual(result['driver_version'], '595.99.02')
        link = result['gpus'][0]['pci']['pci_gpu_link_info']
        self.assertEqual(link['pcie_gen'], {'max_link_gen': '4', 'current_link_gen': '1'})
        self.assertEqual(link['link_widths']['current_link_width'], '16x')

    def test_truncated_xml_is_explicit_instead_of_partial_gpu_information(self):
        with patch('scripts.hardware_report.subprocess.run',
                   return_value=subprocess.CompletedProcess([], 0, 'x' * (1024**2 + 1), '')):
            self.assertIn('capture limit', gpu_info()['error'])
