# Protocol Buffers

messages defined in .proto



## Compiling for Python
protoc --proto_path=$SRC_DIR --python_out=$DST_DIR $SRC_DIR/addressbook.proto

protoc star_data.proto --python_out=.

## Compiling for C++
protoc -I=$SRC_DIR --cpp_out=$DST_DIR $SRC_DIR/addressbook.proto
