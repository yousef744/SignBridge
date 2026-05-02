#!/usr/bin/env python3
"""
SignBridge Pro - Test Suite
Unit tests for backend API and model
"""

import unittest
import json
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, load_model

class TestSignBridgeAPI(unittest.TestCase):
    """Test Flask API endpoints"""

    def setUp(self):
        """Set up test client"""
        self.app = app
        self.client = self.app.test_client()
        self.app.testing = True

    def test_status_endpoint(self):
        """Test /api/status"""
        response = self.client.get('/api/status')
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertIn('model_loaded', data)
        self.assertEqual(data['status'], 'online')

    def test_health_endpoint(self):
        """Test /api/health"""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')

    def test_predict_no_image(self):
        """Test /api/predict without image"""
        response = self.client.post('/api/predict',
                                    data=json.dumps({}),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 400)

        data = json.loads(response.data)
        self.assertFalse(data['success'])

    def test_predict_with_image(self):
        """Test /api/predict with image"""
        # Create a simple test image
        img = Image.new('L', (28, 28), color=128)
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()

        response = self.client.post('/api/predict',
                                    data=json.dumps({'image': f'data:image/png;base64,{img_str}'}),
                                    content_type='application/json')

        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('prediction', data)
        self.assertIn('confidence', data)
        self.assertIn('top3', data)

    def test_batch_predict(self):
        """Test /api/batch_predict"""
        # Create test images
        images = []
        for _ in range(3):
            img = Image.new('L', (28, 28), color=128)
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            images.append(f'data:image/png;base64,{img_str}')

        response = self.client.post('/api/batch_predict',
                                    data=json.dumps({'images': images}),
                                    content_type='application/json')

        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('prediction', data)
        self.assertIn('frames_processed', data)

    def test_save_translation(self):
        """Test /api/save"""
        test_data = {
            'text': 'HELLO',
            'timestamp': '2024-01-01T00:00:00'
        }

        response = self.client.post('/api/save',
                                    data=json.dumps(test_data),
                                    content_type='application/json')

        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('filename', data)

    def test_index_page(self):
        """Test main page loads"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'SignBridge Pro', response.data)


class TestModelFunctions(unittest.TestCase):
    """Test model-related functions"""

    def test_preprocess_image(self):
        """Test image preprocessing"""
        from app import preprocess_image

        # Create test image
        img = Image.new('L', (28, 28), color=128)
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()

        result = preprocess_image(f'data:image/png;base64,{img_str}')

        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (1, 28, 28, 1))
        self.assertTrue(np.all(result >= 0) and np.all(result <= 1))

    def test_label_mapping(self):
        """Test label mapping"""
        from app import ASL_LABELS

        self.assertEqual(len(ASL_LABELS), 27)  # A-Z + space
        self.assertEqual(ASL_LABELS[0], 'A')
        self.assertEqual(ASL_LABELS[25], 'Z')
        self.assertEqual(ASL_LABELS[26], 'space')


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestSignBridgeAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestModelFunctions))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    print("=" * 60)
    print("🧪 SignBridge Pro - Test Suite")
    print("=" * 60)

    success = run_tests()

    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
