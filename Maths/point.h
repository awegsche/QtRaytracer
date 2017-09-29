#ifndef POINT_H
#define POINT_H
#include "constants.h"

class Vector;

class Point
{
public:
    real3 data;
public:
    Point();
    Point(real x, real y, real z);
    Point(const real3 &xyz);

    const Point& operator=(const Vector& v);

    Point &operator += (const Vector& v);

	const real X() const {
        return data.get_x();
	}
	const real Y() const {
        return data.get_y();
	}
	const real Z() const {
        return data.get_z();
	}
};

const Point operator+(const Point& p, const Vector& v);
const Point operator-(const Point& p, const Vector& v);
const Vector operator-(const Point& a, const Point& b);

#endif // POINT_H
