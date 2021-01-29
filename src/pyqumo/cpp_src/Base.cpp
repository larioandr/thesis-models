//
// Created by Andrey Larionov on 28.01.2021.
//

#include "Base.h"
#include <sstream>


// ==========================================================================
// Class Object
// ==========================================================================

Object::Object() = default;
Object::~Object() = default;

std::string Object::toString() const {
    std::stringstream ss;
    ss << "(Object: addr=" << this << ")";
    return ss.str();
}
