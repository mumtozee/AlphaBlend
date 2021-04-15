#include <emmintrin.h>
#include <smmintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  uint16_t magic_;
  uint32_t file_size_;
  uint16_t reserved1_;
  uint16_t reserved2_;
  uint32_t data_offset_;
  uint32_t header_size_;
  uint32_t width_;
  uint32_t height_;
  uint16_t planes_;
  uint16_t bits_per_pixel_;
  uint32_t compression_;
  uint32_t img_data_size_;
  uint32_t h_pix_per_meter_;
  uint32_t v_pix_per_meter_;
  uint32_t colors_used_;
  uint32_t colors_required_;
  uint32_t red_mask_;
  uint32_t green_mask_;
  uint32_t blue_mask_;
  uint32_t alpha_mask_;
} __attribute__((__packed__)) BmpHeader;

typedef struct {
  BmpHeader header_;
  uint8_t* data_;
} BitMap;

BitMap* CreateBMP(uint32_t width, uint32_t height) {
  BitMap* bmp = NULL;
  int bytes_per_pixel = 4;
  uint32_t bytes_per_row = 0;

  bmp = calloc(1, sizeof(BitMap));
  if (bmp == NULL) {
    return NULL;
  }

  bmp->header_.magic_ = 0x4D42;  // BM
  bmp->header_.reserved1_ = 0;
  bmp->header_.reserved2_ = 0;
  bmp->header_.header_size_ = 56;
  bmp->header_.planes_ = 1;
  bmp->header_.compression_ = 0;
  bmp->header_.h_pix_per_meter_ = 0;
  bmp->header_.v_pix_per_meter_ = 0;
  bmp->header_.colors_used_ = 0;
  bmp->header_.colors_required_ = 0;
  bmp->header_.alpha_mask_ = (uint32_t)0xff;          // 00 00 00 FF
  bmp->header_.blue_mask_ = (uint32_t)(0xff << 8);    // 00 00 FF 00
  bmp->header_.green_mask_ = (uint32_t)(0xff << 16);  // 00 FF 00 00
  bmp->header_.red_mask_ = (uint32_t)(0xff << 24);    // FF 00 00 00

  bytes_per_row = width * bytes_per_pixel;

  bmp->header_.width_ = width;
  bmp->header_.height_ = height;
  bmp->header_.bits_per_pixel_ = 32;
  bmp->header_.img_data_size_ = bytes_per_row * height;
  bmp->header_.file_size_ = bmp->header_.img_data_size_ + 70;
  bmp->header_.data_offset_ = 70;

  bmp->data_ = (uint8_t*)calloc(bmp->header_.img_data_size_, sizeof(uint8_t));
  if (bmp->data_ == NULL) {
    free(bmp);
    return NULL;
  }
  return bmp;
}

void DeleteBitMap(BitMap* bmp) {
  if (bmp != NULL) {
    if (bmp->data_ != NULL) {
      free(bmp->data_);
    }
    free(bmp);
  }
}

BitMap* ReadImageFile(const char* filename) {
  BitMap* bmp = calloc(1, sizeof(BitMap));
  if (bmp == NULL) {
    return NULL;
  }

  FILE* f = fopen(filename, "rb");
  if (f == NULL) {
    free(bmp);
    return NULL;
  }

  if (fread(&bmp->header_, sizeof(uint8_t), 70, f) != 70 ||
      bmp->header_.magic_ != 0x4D42) {
    fclose(f);
    free(bmp);
    return NULL;
  }

  bmp->data_ = (uint8_t*)malloc(bmp->header_.width_ * bmp->header_.height_ * 4);
  if (bmp->data_ == NULL) {
    fclose(f);
    free(bmp);
    return NULL;
  }
  if (fread(bmp->data_, sizeof(uint8_t),
            bmp->header_.width_ * bmp->header_.height_ * 4,
            f) != bmp->header_.width_ * bmp->header_.height_ * 4) {
    fclose(f);
    free(bmp->data_);
    free(bmp);
    return NULL;
  }
  fclose(f);
  return bmp;
}

void WriteToImageFile(BitMap* bmp, const char* filename) {
  FILE* f = fopen(filename, "wb");
  if (f == NULL) {
    return;
  }

  if (fwrite(&bmp->header_, sizeof(uint8_t), 70, f) != 70) {
    fclose(f);
    return;
  }
  if (fwrite(bmp->data_, sizeof(uint8_t),
             bmp->header_.width_ * bmp->header_.height_ * 4,
             f) != bmp->header_.width_ * bmp->header_.height_ * 4) {
    fclose(f);
    return;
  }
  fclose(f);
}

uint32_t height(const BitMap* bmp) {
  return bmp->header_.height_;
}

uint32_t width(const BitMap* bmp) {
  return bmp->header_.width_;
}

__m128d ChannelMaskToPackedDouble(__m128i input, __m128i mask) {
  __m128i result_int = _mm_shuffle_epi8(input, mask);
  uint8_t out[16] = {};
  _mm_storeu_si128((__m128i*)out, result_int);
  double tmp[2] = {(double)out[3], (double)out[7]};
  return _mm_loadu_pd(tmp);
}

__m128i AssembleFourPixels(__m128d a, __m128d r, __m128d g, __m128d b) {
  const __m128i alpha_store =
      _mm_setr_epi8(0, 0xff, 0xff, 0xff, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                    0xff, 0xff, 0xff, 0xff, 0xff);
  const __m128i red_store =
      _mm_setr_epi8(0xff, 0, 0xff, 0xff, 0xff, 1, 0xff, 0xff, 0xff, 0xff, 0xff,
                    0xff, 0xff, 0xff, 0xff, 0xff);
  const __m128i green_store =
      _mm_setr_epi8(0xff, 0xff, 0, 0xff, 0xff, 0xff, 1, 0xff, 0xff, 0xff, 0xff,
                    0xff, 0xff, 0xff, 0xff, 0xff);
  const __m128i blue_store =
      _mm_setr_epi8(0xff, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 1, 0xff, 0xff, 0xff,
                    0xff, 0xff, 0xff, 0xff, 0xff);
  __m128i a_i = _mm_cvttpd_epi32(a);
  a_i = _mm_packus_epi32(a_i, a_i);
  a_i = _mm_packus_epi16(a_i, a_i);
  a_i = _mm_shuffle_epi8(a_i, alpha_store);
  __m128i r_i = _mm_cvttpd_epi32(r);
  r_i = _mm_packus_epi32(r_i, r_i);
  r_i = _mm_packus_epi16(r_i, r_i);
  r_i = _mm_shuffle_epi8(r_i, red_store);
  __m128i g_i = _mm_cvttpd_epi32(g);
  g_i = _mm_packus_epi32(g_i, g_i);
  g_i = _mm_packus_epi16(g_i, g_i);
  g_i = _mm_shuffle_epi8(g_i, green_store);
  __m128i b_i = _mm_cvttpd_epi32(b);
  b_i = _mm_packus_epi32(b_i, b_i);
  b_i = _mm_packus_epi16(b_i, b_i);
  b_i = _mm_shuffle_epi8(b_i, blue_store);

  __m128i result = _mm_xor_si128(a_i, r_i);
  result = _mm_xor_si128(result, _mm_xor_si128(g_i, b_i));
  return result;
}

__m128i AlphaBlend(__m128i bg, __m128i fg) {
  const __m128i get_alpha =
      _mm_setr_epi8(0xff, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 4, 0xff, 0xff, 0xff,
                    0xff, 0xff, 0xff, 0xff, 0xff);
  const __m128i get_red =
      _mm_setr_epi8(0xff, 0xff, 0xff, 1, 0xff, 0xff, 0xff, 5, 0xff, 0xff, 0xff,
                    0xff, 0xff, 0xff, 0xff, 0xff);
  const __m128i get_green =
      _mm_setr_epi8(0xff, 0xff, 0xff, 2, 0xff, 0xff, 0xff, 6, 0xff, 0xff, 0xff,
                    0xff, 0xff, 0xff, 0xff, 0xff);
  const __m128i get_blue =
      _mm_setr_epi8(0xff, 0xff, 0xff, 3, 0xff, 0xff, 0xff, 7, 0xff, 0xff, 0xff,
                    0xff, 0xff, 0xff, 0xff, 0xff);

  const double m255f[2] = {255.0, 255.0};
  const double onesf[2] = {1.0, 1.0};
  const __m128d m255 = _mm_loadu_pd(m255f);
  const __m128d ones = _mm_loadu_pd(onesf);

  __m128d fg_alpha = ChannelMaskToPackedDouble(fg, get_alpha);
  fg_alpha = _mm_div_pd(fg_alpha, m255);
  const __m128d inv_alpha = _mm_sub_pd(ones, fg_alpha);

  __m128d bg_alpha = ChannelMaskToPackedDouble(bg, get_alpha);
  bg_alpha = _mm_div_pd(bg_alpha, m255);

  __m128d res_alpha = _mm_add_pd(_mm_mul_pd(bg_alpha, inv_alpha), fg_alpha);

  __m128d fg_green = ChannelMaskToPackedDouble(fg, get_green);
  __m128d fg_red = ChannelMaskToPackedDouble(fg, get_red);
  __m128d fg_blue = ChannelMaskToPackedDouble(fg, get_blue);
  __m128d bg_green = ChannelMaskToPackedDouble(bg, get_green);
  __m128d bg_red = ChannelMaskToPackedDouble(bg, get_red);
  __m128d bg_blue = ChannelMaskToPackedDouble(bg, get_blue);

  __m128d res_red =
      _mm_add_pd(_mm_mul_pd(bg_red, inv_alpha), _mm_mul_pd(fg_red, fg_alpha));
  res_red = _mm_div_pd(res_red, res_alpha);

  __m128d res_green = _mm_add_pd(_mm_mul_pd(bg_green, inv_alpha),
                                 _mm_mul_pd(fg_green, fg_alpha));
  res_green = _mm_div_pd(res_green, res_alpha);

  __m128d res_blue =
      _mm_add_pd(_mm_mul_pd(bg_blue, inv_alpha), _mm_mul_pd(fg_blue, fg_alpha));
  res_blue = _mm_div_pd(res_blue, res_alpha);
  res_alpha = _mm_mul_pd(res_alpha, m255);
  return AssembleFourPixels(res_alpha, res_red, res_green, res_blue);
}

BitMap* BlendImages(const BitMap* bg, const BitMap* fg, uint32_t x,
                    uint32_t y) {
  BitMap* result = CreateBMP(width(bg), height(bg));
  result->header_.compression_ = bg->header_.compression_;
  result->header_.h_pix_per_meter_ = bg->header_.h_pix_per_meter_;
  result->header_.v_pix_per_meter_ = bg->header_.v_pix_per_meter_;

  for (uint32_t i = 0; i < height(fg); ++i) {
    for (uint32_t j = 0; j < width(fg); j += 2) {
      uint32_t bg_pos = ((i + y) * width(bg) + j + x) << 2;
      uint32_t fg_pos = (i * width(fg) + j) << 2;

      __m128i under = _mm_loadu_si128((const __m128i*)(bg->data_ + bg_pos));
      __m128i over = _mm_loadu_si128((const __m128i*)(fg->data_ + fg_pos));
      __m128i blend4 = AlphaBlend(under, over);
      _mm_storeu_si128((__m128i*)(result->data_ + bg_pos), blend4);
    }
  }

  return result;
}

void PrintImage(BitMap* bmp) {
  printf("Offset %d\n", bmp->header_.data_offset_);
  printf("Header Size %d\n", bmp->header_.header_size_);
  printf("bits per pixel %x\n", bmp->header_.bits_per_pixel_);
  printf("compression %x\n", bmp->header_.compression_);
  printf("ImageSize %d\n", bmp->header_.img_data_size_);
  printf("h resolution %x\n", bmp->header_.h_pix_per_meter_);
  printf("v resulution %x\n", bmp->header_.v_pix_per_meter_);
  printf("Colors used %x\n", bmp->header_.colors_used_);
  printf("Colors required %x\n", bmp->header_.colors_required_);
  printf("Red Mask %x\n", bmp->header_.red_mask_);
  printf("Green Mask %x\n", bmp->header_.green_mask_);
  printf("Blue Mask %x\n", bmp->header_.blue_mask_);
  printf("Alpha Mask %x\n", bmp->header_.alpha_mask_);
}

void PrintPixel(const BitMap* bmp, uint32_t x, uint32_t y) {
  uint8_t alpha = bmp->data_[y * width(bmp) * 4 + x * 4];
  uint8_t r = bmp->data_[y * width(bmp) * 4 + x * 4 + 1];
  uint8_t g = bmp->data_[y * width(bmp) * 4 + x * 4 + 2];
  uint8_t b = bmp->data_[y * width(bmp) * 4 + x * 4 + 3];
  printf("A:%d R:%d G:%d B:%d\n", alpha, r, g, b);
}

int main(int argc, char* argv[]) {
  char* src_img_1 = argv[1];
  char* src_img_2 = argv[2];
  char* dest_img = argv[3];
  // we assume foreground size <= background size
  BitMap* picture_a = ReadImageFile(src_img_1);
  BitMap* picture_b = ReadImageFile(src_img_2);
  BitMap* blended_pic = BlendImages(picture_a, picture_b, 0, 0);
  WriteToImageFile(blended_pic, dest_img);
  DeleteBitMap(picture_a);
  DeleteBitMap(picture_b);
  DeleteBitMap(blended_pic);
  return 0;
}