#!/bin/bash
set -e

echo "🚀 Publishing Memory AI SDKs to PyPI and npm"
echo "=============================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -d "sdk/python" ] || [ ! -d "sdk/typescript" ]; then
    echo -e "${RED}❌ Error: Must run from project root directory${NC}"
    exit 1
fi

# Python SDK
echo -e "\n${BLUE}📦 Publishing Python SDK to PyPI...${NC}"
cd sdk/python

echo "🧹 Cleaning old builds..."
rm -rf dist/ build/ *.egg-info

echo "📄 Copying LICENSE..."
if [ -f "../../LICENSE" ]; then
    cp ../../LICENSE ./LICENSE
else
    echo -e "${YELLOW}⚠️  Warning: LICENSE file not found${NC}"
fi

echo "🔨 Building package..."
python -m build

echo "🧪 Checking package..."
python -m twine check dist/*

echo "📤 Uploading to PyPI..."
echo -e "${YELLOW}You will be prompted for PyPI credentials:${NC}"
echo "  Username: __token__"
echo "  Password: pypi-YOUR_API_TOKEN"
python -m twine upload dist/*

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Python SDK published to PyPI!${NC}"
    PYTHON_SUCCESS=true
else
    echo -e "${RED}❌ Python SDK publishing failed${NC}"
    PYTHON_SUCCESS=false
fi

# TypeScript SDK
echo -e "\n${BLUE}📦 Publishing TypeScript SDK to npm...${NC}"
cd ../typescript

echo "🧹 Cleaning old builds..."
rm -rf dist/

echo "📄 Copying LICENSE..."
if [ -f "../../LICENSE" ]; then
    cp ../../LICENSE ./LICENSE
else
    echo -e "${YELLOW}⚠️  Warning: LICENSE file not found${NC}"
fi

echo "📦 Installing dependencies..."
npm install

echo "🔨 Building package..."
npm run build

echo "📤 Publishing to npm..."
echo -e "${YELLOW}Make sure you're logged in to npm (run 'npm login' if needed)${NC}"
npm publish --access public

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ TypeScript SDK published to npm!${NC}"
    TYPESCRIPT_SUCCESS=true
else
    echo -e "${RED}❌ TypeScript SDK publishing failed${NC}"
    TYPESCRIPT_SUCCESS=false
fi

# Summary
echo -e "\n${BLUE}=============================================="
echo "📊 Publishing Summary"
echo -e "===============================================${NC}\n"

if [ "$PYTHON_SUCCESS" = true ]; then
    echo -e "${GREEN}✅ Python SDK: Published${NC}"
    echo "   📦 PyPI: https://pypi.org/project/memory-ai-sdk/"
    echo "   💻 Install: pip install memory-ai-sdk"
else
    echo -e "${RED}❌ Python SDK: Failed${NC}"
fi

echo ""

if [ "$TYPESCRIPT_SUCCESS" = true ]; then
    echo -e "${GREEN}✅ TypeScript SDK: Published${NC}"
    echo "   📦 npm: https://www.npmjs.com/package/memory-ai-sdk"
    echo "   💻 Install: npm install memory-ai-sdk"
else
    echo -e "${RED}❌ TypeScript SDK: Failed${NC}"
fi

echo ""

if [ "$PYTHON_SUCCESS" = true ] && [ "$TYPESCRIPT_SUCCESS" = true ]; then
    echo -e "${GREEN}🎉 All SDKs published successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Verify installations work"
    echo "  2. Update main README with installation instructions"
    echo "  3. Create GitHub release (git tag v1.0.0)"
    echo "  4. Announce on Twitter/Discord/Reddit"
    echo "  5. Monitor downloads and issues"
    exit 0
else
    echo -e "${YELLOW}⚠️  Some SDKs failed to publish. Check errors above.${NC}"
    exit 1
fi

