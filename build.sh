#!/bin/bash
# Build script for Sizing API - Supports both CPU and GPU builds

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Display usage
usage() {
    cat << EOF
Sizing API Docker Build Script

Usage: ./build.sh [OPTIONS]

OPTIONS:
    -m, --mode MODE      Build mode: cpu, gpu, or both (default: cpu)
    -t, --tag TAG        Custom tag for the image (default: latest)
    -p, --push           Push images to registry after build
    -c, --clean          Clean build (remove cache and rebuild)
    -h, --help           Display this help message

EXAMPLES:
    ./build.sh                      # Build CPU-only version
    ./build.sh -m gpu               # Build GPU version only
    ./build.sh -m both              # Build both CPU and GPU versions
    ./build.sh -m gpu -p            # Build GPU version and push to registry
    ./build.sh -m both -c           # Clean build of both versions

REQUIREMENTS FOR GPU:
    - NVIDIA Docker runtime (nvidia-docker2)
    - CUDA-capable GPU
    - GPU drivers installed on host

EOF
    exit 0
}

# Default values
BUILD_MODE="cpu"
IMAGE_TAG="latest"
PUSH_IMAGE=false
CLEAN_BUILD=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            BUILD_MODE="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -p|--push)
            PUSH_IMAGE=true
            shift
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate build mode
if [[ ! "$BUILD_MODE" =~ ^(cpu|gpu|both)$ ]]; then
    print_error "Invalid build mode: $BUILD_MODE. Must be cpu, gpu, or both."
    exit 1
fi

# Clean build if requested
if [ "$CLEAN_BUILD" = true ]; then
    print_info "Cleaning Docker build cache..."
    docker builder prune -f
    print_success "Build cache cleaned"
fi

# Build CPU version
build_cpu() {
    print_info "Building CPU-only version..."
    docker build \
        --build-arg CUDA_ENABLED=cpu \
        --build-arg PYTHON_VERSION=3.8 \
        -t sizing-api:cpu-${IMAGE_TAG} \
        -t sizing-api:cpu-latest \
        -f Dockerfile \
        .

    if [ $? -eq 0 ]; then
        print_success "CPU version built successfully"

        # Get image size
        IMAGE_SIZE=$(docker images sizing-api:cpu-${IMAGE_TAG} --format "{{.Size}}")
        print_info "Image size: $IMAGE_SIZE"

        if [ "$PUSH_IMAGE" = true ]; then
            print_info "Pushing CPU image to registry..."
            docker push sizing-api:cpu-${IMAGE_TAG}
            docker push sizing-api:cpu-latest
            print_success "CPU image pushed successfully"
        fi
    else
        print_error "CPU build failed"
        exit 1
    fi
}

# Build GPU version
build_gpu() {
    # Check for NVIDIA Docker runtime
    if ! docker info 2>/dev/null | grep -q nvidia; then
        print_warning "NVIDIA Docker runtime not detected. GPU build may fail at runtime."
        print_warning "Install nvidia-docker2 for GPU support: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi

    print_info "Building GPU-enabled version..."
    docker build \
        --build-arg CUDA_ENABLED=gpu \
        --build-arg PYTHON_VERSION=3.8 \
        -t sizing-api:gpu-${IMAGE_TAG} \
        -t sizing-api:gpu-latest \
        -f Dockerfile \
        .

    if [ $? -eq 0 ]; then
        print_success "GPU version built successfully"

        # Get image size
        IMAGE_SIZE=$(docker images sizing-api:gpu-${IMAGE_TAG} --format "{{.Size}}")
        print_info "Image size: $IMAGE_SIZE"

        if [ "$PUSH_IMAGE" = true ]; then
            print_info "Pushing GPU image to registry..."
            docker push sizing-api:gpu-${IMAGE_TAG}
            docker push sizing-api:gpu-latest
            print_success "GPU image pushed successfully"
        fi
    else
        print_error "GPU build failed"
        exit 1
    fi
}

# Main build logic
print_info "Starting Sizing API build process..."
print_info "Build mode: $BUILD_MODE"
print_info "Image tag: $IMAGE_TAG"
print_info "Push to registry: $PUSH_IMAGE"

case $BUILD_MODE in
    cpu)
        build_cpu
        ;;
    gpu)
        build_gpu
        ;;
    both)
        build_cpu
        echo ""
        build_gpu
        ;;
esac

echo ""
print_success "Build process completed!"
print_info "To run the container:"
if [ "$BUILD_MODE" = "cpu" ] || [ "$BUILD_MODE" = "both" ]; then
    echo "  CPU: docker run -p 8000:80 sizing-api:cpu-${IMAGE_TAG}"
fi
if [ "$BUILD_MODE" = "gpu" ] || [ "$BUILD_MODE" = "both" ]; then
    echo "  GPU: docker run --gpus all -p 8000:80 sizing-api:gpu-${IMAGE_TAG}"
fi

print_info "Or use docker-compose:"
if [ "$BUILD_MODE" = "cpu" ] || [ "$BUILD_MODE" = "both" ]; then
    echo "  CPU: docker-compose --profile cpu up"
fi
if [ "$BUILD_MODE" = "gpu" ] || [ "$BUILD_MODE" = "both" ]; then
    echo "  GPU: docker-compose --profile gpu up"
fi
