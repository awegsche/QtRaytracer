#ifndef __TRIANGLE__
#define __TRIANGLE__


// 	Copyright (C) Kevin Suffern 2000-2007.
//	This C++ code is for non-commercial purposes only.
//	This C++ code is licensed under the GNU General Public License Version 2.
//	See the file COPYING.txt for the full license.


// this is the triangle discussed in Section 19.3

#include "geometricobject.h"

class Triangle: public GeometricObject {	
	public:
	
		Triangle(void);   												
				
        Triangle(const Point& a, const Point& b, const Point& c);

		virtual Triangle* 												
		clone(void) const;
	
		Triangle(const Triangle& triangle); 							

		~Triangle(void);   											

		Triangle& 														
		operator= (const Triangle& rhs);
		
		virtual BBox															
		get_bounding_box(void);

		void 
		compute_normal(void);
		
		virtual bool 
		hit(const Ray& ray, real& tmin, ShadeRec& sr) const;		 					 
		
		virtual bool 																						 
		shadow_hit(const Ray& ray, real& tmin) const; 
		
	private:
	
        Point	v0, v1, v2;
		Normal	normal;
};

#endif

