#!/bin/bash
# Run all tests
set -e

echo "Running profit lock tests..."
python test_profit_lock.py

echo ""
echo "✅ All tests passed!"
