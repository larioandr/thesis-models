#ifndef BASE_H
#define BASE_H

#include <string>
#include <sstream>

#ifdef VERBOSE
#define debug(...) printf(__VA_ARGS__)
#else
#define debug(...) /* nop */
#endif


class Object {
public:
    Object();
    virtual ~Object();

    virtual std::string toString() const;
};


template<typename T>
std::string toString(const std::vector<T>& array, const std::string& delim = ", ") {
    std::stringstream ss;
    ss << "[";
    if (array.size() > 0) {
        ss << array[0];
        for (unsigned i = 1; i < array.size(); i++) {
            ss << delim << array[i];
        }
    }
    ss << "]";
    return ss.str();
}

#endif //BASE_H
