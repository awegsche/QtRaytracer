#pragma once

#ifdef WCUDA

template <class CUDADeviceObject>
class CUDAHostObject
{
protected:
	CUDADeviceObject* device_ptr;
public:
	CUDAHostObject<CUDADeviceObject>();
	~CUDAHostObject();

	virtual CUDADeviceObject* get_device_ptr();
};


#endif // WCUDA

template<class CUDADeviceObject>
inline CUDAHostObject<CUDADeviceObject>::CUDAHostObject()
{
}

template<class CUDADeviceObject>
inline CUDAHostObject<CUDADeviceObject>::~CUDAHostObject()
{
}

template<class CUDADeviceObject>
inline CUDADeviceObject * CUDAHostObject<CUDADeviceObject>::get_device_ptr()
{
	return NULL;
}
