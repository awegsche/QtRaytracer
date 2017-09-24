#ifndef MCWATERBLOCK_H
#define MCWATERBLOCK_H

#include "mcblock.h"
#include "mcstandardblock.h"

class MCWaterBlock : public MCStandardBlock
{
public:
    MCWaterBlock();
    MCWaterBlock(Material* material);
};

#endif // MCWATERBLOCK_H
