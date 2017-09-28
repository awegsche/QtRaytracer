#ifndef POINT_H
#define POINT_H
#include "constants.h"

class Vector;

class Point
{
public:
	real4 data;
public:
    Point();
    Point(real x, real y, real z);
	Point(const real4 &xyzw);

    const Point& operator=(const Vector& v);

    Point &operator += (const Vector& v);

	const real X() const {
		return data[3];
	}
	const real Y() const {
		return data[2];
	}
	const real Z() const {
		return data[1];
	}
};

const Point operator+(const Point& p, const Vector& v);
const Point operator-(const Point& p, const Vector& v);
const Vector operator-(const Point& a, const Point& b);

#endif // POINT_H
