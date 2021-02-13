#ifndef _SRC_SHOW_PP_DATA_H
#define _SRC_SHOW_PP_DATA_H

#include "ppi.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cnpy.h"

using namespace cv;
using namespace cnpy;

class ShowPPData {
public:
    ShowPPData();
    ~ShowPPData();

    void Run();
    void SetData(double_t *p_data, uint32_t data_len);
    void SetShowData(uint32_t start_index, uint32_t show_len);
    void Zoom(double zoom_ratio);
    void SetDataSafeFilter();
    void RefreshImage();
    void RefreshImageDynamic();
    void ShowActiveData();
    void SetPPI(uint32_t ppi);
    void DataRange(uint32_t ppi, double &max_value, double &min_value);
    void MouseHandler(int event, int x, int y, int flags);
    static void MouseHandlerEntry(int event, int x, int y, int flags, void *param);

private:
    Size image_size_;
    Rect price_window_;
    Rect vol_window_;
    Mat show_image_;
    Mat show_image_dynamic_;
    bool show_ppi_[PPI_NUM];
    bool ppi_is_vol_[PPI_NUM];
    double_t *data_ = NULL;
    uint32_t data_len_ = 0;
    uint32_t show_start_index_ = 0;
    uint32_t show_len_setting_ = 0;
    uint32_t show_len_;
    double close_ratio_;
    double vol_ratio_;
    int32_t  active_data_index_ = -1;
    const char *window_name_;
    Point mouse_pt_;
    double data_unit_width;
    double max_close;
    double min_close;
    double max_vol;
    double min_vol;
};

#define PP_DATA(_index, _ppi) data_[(_index) * PPI_NUM + (_ppi)]
#define PP_DATA_P(_index, _ppi) (data_ + (_index) * PPI_NUM + (_ppi))

enum ZoomOptions {
    ZOOM_IN = 0,
    ZOOM_OUT
};

#endif // _SRC_SHOW_PP_DATA_H


