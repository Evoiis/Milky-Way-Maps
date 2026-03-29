# Protocol Buffers

messages defined in .proto


## Python
Compiling:
`protoc star_data.proto --python_out=star_data_pb2/`

Install python package:
`pip install -e .`

-e for editable, if developing


## Compiling for C++
Compiled using Cmake in visualization/CmakeLists.txt
