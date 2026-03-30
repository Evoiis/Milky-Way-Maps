#ifndef VIS_H
#define VIS_H

#include "star_data.hpp"
#include "camera.hpp"
#include "bloom_pipeline.hpp"

class Visualization{
public:

    Visualization(std::shared_ptr<SharedStars> shared_stars_ptr, Camera camera, BloomPipeline bp);

    void run();

private:

    void render_loop();

};

#endif
