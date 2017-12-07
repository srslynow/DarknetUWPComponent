#include "DarknetComponent.h"

using namespace DarknetComponentns;
using namespace concurrency;
using namespace Platform;
using namespace Windows::Storage;
using namespace Windows::Storage::Streams;

DarknetComponent::DarknetComponent()
{
	network = parse_network_cfg("Assets/tiny-yolo.cfg");
	load_weights(&network, "Assets/tiny-yolo.weights");
	//network = parse_network_cfg("Assets/coco_yolo_mobilenet.cfg");
	//load_weights(&network, "Assets/mobilenet_yolo_coco.weights");
	set_batch_network(&network, 1);
	//names = get_labels("coco.names");
}

Windows::Foundation::Collections::IVector<Rect>^ DarknetComponentns::DarknetComponent::testFunction()
{
	throw ref new Platform::NotImplementedException();
	// TODO: insert return statement here
}

Windows::Foundation::Collections::IVector<Rect>^ DarknetComponentns::DarknetComponent::predictImage(const Platform::Array<unsigned char>^ imageData)
{
	image im = load_image_from_mem(imageData->Data, imageData->Length);
	layer l = network.layers[network.n - 1];
	int j;

	box *boxes = (box*)calloc(l.w*l.h*l.n, sizeof(box));
	float **probs = (float**)calloc(l.w*l.h*l.n, sizeof(float *));
	for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float*)calloc(l.classes + 1, sizeof(float *));
	float **masks = 0;
	if (l.coords > 4) {
		masks = (float**)calloc(l.w*l.h*l.n, sizeof(float*));
		for (j = 0; j < l.w*l.h*l.n; ++j) masks[j] = (float*)calloc(l.coords - 4, sizeof(float *));
	}

	image sized = letterbox_image(im, network.w, network.h);
	float *X = sized.data;

	network_predict(network, X);

	get_region_boxes(l, im.w, im.h, network.w, network.h, 0.5, probs, boxes, masks, 0, 0, 0.5, 1);
	do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, 0.3);

	auto vec = ref new Platform::Collections::Vector<Rect>;
	int num_detections = l.w*l.h*l.n;
	for (int i = 0; i < num_detections; ++i) {
		int class_id = max_index(probs[i], l.classes);
		float prob = probs[i][class_id];
		if (prob > 0.0) {
			box b = boxes[i];
			Rect newRect;
			newRect.left = (b.x - b.w / 2.)*im.w;
			newRect.right = (b.x + b.w / 2.)*im.w;
			newRect.top = (b.y - b.h / 2.)*im.h;
			newRect.bottom = (b.y + b.h / 2.)*im.h;
			newRect.prob = prob;
			newRect.classId = class_id;
			vec->Append(newRect);
		}
	}
	//throw ref new Platform::NotImplementedException();
	return vec;
}

image DarknetComponentns::DarknetComponent::load_image_from_mem(unsigned char * imageData, int len)
{
	int w, h, c;
	unsigned char *data = stbi_load_from_memory(imageData, len, &w, &h, &c, 3);
	int i, j, k;
	image im = make_image(w, h, c);
	for (k = 0; k < c; ++k) {
		for (j = 0; j < h; ++j) {
			for (i = 0; i < w; ++i) {
				int dst_index = i + w*j + w*h*k;
				int src_index = k + c*i + c*w*j;
				im.data[dst_index] = (float)data[src_index] / 255.;
			}
		}
	}
	free(data);
	return im;
}
