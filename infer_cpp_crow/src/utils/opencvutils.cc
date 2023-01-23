#include "opencvutils.h"

// Resize an image to a given size to
cv::Mat __resize_to_a_size(cv::Mat image, int new_height, int new_width)
{

  // get original image size
  int org_image_height = image.rows;
  int org_image_width = image.cols;

  // get image area and resized image area
  float img_area = float(org_image_height * org_image_width);
  float new_area = float(new_height * new_width);

  // resize
  cv::Mat image_scaled;
  cv::Size scale(new_width, new_height);

  if (new_area >= img_area)
  {
    cv::resize(image, image_scaled, scale, 0, 0, cv::INTER_LANCZOS4);
  }
  else
  {
    cv::resize(image, image_scaled, scale, 0, 0, cv::INTER_AREA);
  }

  return image_scaled;
}

cv::Mat preprocess(cv::Mat image, int new_height, int new_width)
{
  cv::Mat image_proc = image.clone();
  cv::cvtColor(image_proc, image_proc, cv::COLOR_BGR2RGB);

  // Resize image, not doing it in here as it happens on graph with tensor
  /* image_proc = __resize_to_a_size(image_proc, new_height, new_width); */
  return image_proc;
}
