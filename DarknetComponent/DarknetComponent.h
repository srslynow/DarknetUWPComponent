#pragma once

#include <vector>
#include <collection.h>
//#include <StorageProvider.h>
#include <ppltasks.h>

extern "C" {
#include "darknet.h"
#include "data.h"
#include "stb_image.h"
}

namespace DarknetComponentns
{
	public value struct Rect
	{
		float left, right, top, bottom, prob;
		int classId;
	};

	bool operator ==(const Rect& x, const Rect& y) {
		return std::tie(x.left, x.right, x.top, x.bottom) < std::tie(y.left, y.right, y.top, y.bottom);
	}

    public ref class DarknetComponent sealed
    {
    public:
		DarknetComponent();
		Windows::Foundation::Collections::IVector<Rect>^ testFunction();
		Windows::Foundation::Collections::IVector<Rect>^ predictImage(const Platform::Array<unsigned char>^ imageData);

	private:
		image load_image_from_mem(unsigned char * imageData, int len);
		network network;
    };
}
