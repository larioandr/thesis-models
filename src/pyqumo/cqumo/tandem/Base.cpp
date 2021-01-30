/**
 * @author Andrey Larionov
 */
#include "Base.h"
#include <sstream>

namespace cqumo {

Object::Object() = default;

Object::~Object() = default;

std::string Object::toString() const {
    std::stringstream ss;
    ss << "(Object: addr=" << this << ")";
    return ss.str();
}

}
