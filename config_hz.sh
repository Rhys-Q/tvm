# 获取当前脚本的绝对路径
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
echo "当前脚本路径: $SCRIPT_DIR"
mkdir -p $SCRIPT_DIR/build
cp $SCRIPT_DIR/cmake/config.cmake  $SCRIPT_DIR/build/config.cmake