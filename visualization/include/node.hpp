#ifndef NODE_HPP
#define NODE_HPP

#include <iostream>
#include <memory>

#include <zmq.hpp>
#include <google/protobuf/message.h>

#include "star_data.pb.h"
#include "star_data.hpp"

class Node{
public:

    Node(std::shared_ptr<SharedStars> shared_stars_ptr, int port);

    void request_gaia_data();



private:


};

#endif
