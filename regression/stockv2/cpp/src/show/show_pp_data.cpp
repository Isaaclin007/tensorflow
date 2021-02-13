
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <gflags/gflags.h>
#include "show_pp_data.h"
#include "VCAImage.h"
#include "VCALogManage.h"

ShowPPData::ShowPPData()
{
    image_size_.width = 1800;
    image_size_.height = 900;
    uint32_t boder_size = 10;
    double price_ratio = 0.8;
    uint32_t price_height = image_size_.height * price_ratio;
    uint32_t vol_height = image_size_.height - price_height;
    price_window_.x = boder_size;
    price_window_.width = image_size_.width - boder_size * 2;
    price_window_.y = boder_size + 30;
    price_window_.height = price_height - price_window_.y - boder_size;

    vol_window_.x = boder_size;
    vol_window_.width = image_size_.width - boder_size * 2;
    vol_window_.y = price_height;
    vol_window_.height = vol_height - boder_size - 30;

    CreateAImage(show_image_, image_size_, 3);
//    ShowTagRMCT(show_image_, price_window_);
//    ShowTagRMCT(show_image_, vol_window_);


    memset(show_ppi_, 0, sizeof(show_ppi_));
    memset(ppi_is_vol_, 0, sizeof(ppi_is_vol_));
    ppi_is_vol_[PPI_vol] = true;
    ppi_is_vol_[PPI_vol_5] = true;
    ppi_is_vol_[PPI_vol_10_avg] = true;
    ppi_is_vol_[PPI_vol_30_avg] = true;
    ppi_is_vol_[PPI_vol_100_avg] = true;

    window_name_ = "image";
    mouse_pt_.x = 0;
    mouse_pt_.y = 0;
    cvNamedWindow(window_name_,CV_WINDOW_AUTOSIZE);
    cvSetMouseCallback(window_name_, MouseHandlerEntry, this);
}

ShowPPData::~ShowPPData()
{

}

void ShowPPData::Run()
{
    int key_value;
    while (1)
    {
        imshow(window_name_, show_image_dynamic_);
        key_value = waitKey(-1);
        // printf("key_value:%d\n", key_value);
        switch (key_value) {
        case 27:
            return;
        case 83:
            if (show_start_index_ > 0)
            {
                show_start_index_--;
            }
            break;
        case 81:
            if ((show_start_index_ + show_len_) < data_len_)
            {
                show_start_index_++;
            }
            break;
        case 45:
            Zoom(1.1);
            break;
        case 61:
            Zoom(1.0/1.1);
            break;
        default:
            break;
        }
        RefreshImage();
    }
    return;
}

void ShowPPData::Zoom(double zoom_ratio)
{
    uint32_t refer_index, temp_show_len;
    double refer_pos_ratio;
    if (active_data_index_ == -1)
    {
        refer_index = show_start_index_ + show_len_ / 2;
    }
    else
    {
        refer_index = active_data_index_;
    }
    refer_pos_ratio = (double)(refer_index - show_start_index_) / show_len_;

    temp_show_len = show_len_ * zoom_ratio;
    show_len_ = temp_show_len == show_len_? temp_show_len + 1 : temp_show_len;
    int32_t temp_index;
    temp_index = refer_index - show_len_ * refer_pos_ratio;
    if (temp_index < 0)
    {
        temp_index = 0;
    }
    show_start_index_ = temp_index;
}

void ShowPPData::SetDataSafeFilter()
{
    if (show_len_ > data_len_)
    {
        show_len_ = data_len_;
        show_start_index_ = 0;
        active_data_index_ = -1;
    }
    if ((show_start_index_ + show_len_) > data_len_)
    {
        show_start_index_ = data_len_ - show_len_;
        active_data_index_ = -1;
    }
    if (active_data_index_ < show_start_index_)
    {
        active_data_index_ = -1;
    }
    if (active_data_index_ >= show_start_index_ + show_len_)
    {
        active_data_index_ = -1;
    }
}

void ShowPPData::SetData(double_t *p_data, uint32_t data_len)
{
    PRINT_CODE;
    data_ = p_data;
    data_len_ = data_len;
    if (show_len_setting_ == 0)
    {
        show_len_ = data_len_;
    }
    else
    {
        show_len_ = show_len_setting_;
    }
    RefreshImage();
}

void ShowPPData::SetShowData(uint32_t start_index, uint32_t show_len)
{
    show_start_index_ = start_index;
    show_len_ = show_len;
    RefreshImage();
}

void ShowPPData::DataRange(uint32_t ppi, double &max_value, double &min_value)
{
    PRINT_CODE;
    double _max = PP_DATA(show_start_index_, ppi);
    double _min = _max;
    for (int i = 0; i < show_len_; i++)
    {
        double temp_value = PP_DATA(show_start_index_ + i, ppi);
        if (_max < temp_value)
        {
            _max = temp_value;
        }
        if (_min > temp_value)
        {
            _min = temp_value;
        }
    }
    max_value = _max;
    min_value = _min;
}

void ShowPPData::MouseHandler(int event, int x, int y, int flags)
{
    switch (event) {
    case CV_EVENT_MOUSEMOVE:
        mouse_pt_.x = x;
        mouse_pt_.y = y;
        RefreshImageDynamic();
        imshow(window_name_, show_image_dynamic_);
        break;
    default:
        break;
    }
    // printf("event:%d, flags:%d (%d, %d)\n", event, flags, x, y);
}

void ShowPPData::MouseHandlerEntry(int event, int x, int y, int flags, void *param)
{
    ShowPPData *p_object=(ShowPPData *)param;
    p_object->MouseHandler(event, x, y, flags);
}

void ShowPPData::RefreshImage()
{
    PRINT_CODE;
    MatSet(show_image_, g_black);

    SetDataSafeFilter();
    data_unit_width = (double)price_window_.width / (show_len_ + 1);
    DataRange(PPI_close, max_close, min_close);
    close_ratio_ = price_window_.height / (max_close - min_close);

    DataRange(PPI_vol, max_vol, min_vol);
    vol_ratio_ = vol_window_.height / max_vol;

    Point pt1, pt2;
    Rect vol;
    for (int i = 0; i < show_len_; i++)
    {
        pt2 = pt1;
        pt1.x = price_window_.x + price_window_.width - i * data_unit_width - data_unit_width;
        pt1.y = price_window_.y + price_window_.height - (PP_DATA(i + show_start_index_, PPI_close) - min_close) * close_ratio_;
        if (i > 0)
        {
            ShowLine(show_image_, pt1, pt2);
        }

        vol.height = PP_DATA(i + show_start_index_, PPI_vol) * vol_ratio_;
        vol.y = vol_window_.y + vol_window_.height - vol.height;
        vol.width = (int)(data_unit_width * 0.8);
        vol.x = pt1.x - vol.width / 2;
        if (vol.width <= 0)
        {
            ShowLine(show_image_, Point(vol.x, vol.y), Point(vol.x, vol.y + vol.height), g_green);
        }
        else
        {
            rectangle(show_image_, vol, g_green, -1, CV_AA);
        }
    }
    char str_buffer[128];
    Point text_pt;
    sprintf(str_buffer, "%06u", (uint32_t)PP_DATA(0, PPI_ts_code));
    text_pt.x = price_window_.x + price_window_.width / 2 - 50;
    text_pt.y = 30;
    ShowText(show_image_, text_pt, str_buffer, g_white, 0.6, 2);

    sprintf(str_buffer, "%.2f", min_close);
    text_pt.x = price_window_.x + 20;
    text_pt.y = price_window_.y + price_window_.height;
    ShowLine(show_image_, Point(0, text_pt.y), text_pt, g_red);
    text_pt.x += 5;
    text_pt.y += 5;
    ShowText(show_image_, text_pt, str_buffer, g_red, 0.5, 1);

    sprintf(str_buffer, "%.2f", max_close);
    text_pt.x = price_window_.x + 20;
    text_pt.y = price_window_.y + price_window_.height - (max_close - min_close) * close_ratio_;
    ShowLine(show_image_, Point(0, text_pt.y), text_pt, g_red);
    text_pt.x += 5;
    text_pt.y += 5;
    ShowText(show_image_, text_pt, str_buffer, g_red, 0.5, 1);

    uint32_t date = PP_DATA(show_start_index_, PPI_trade_date);
    sprintf(str_buffer, "%04u-%02u-%02u", date / 10000, (date % 10000) / 100, date % 100);
    text_pt.x = vol_window_.x + vol_window_.width - 110;
    text_pt.y = vol_window_.y + vol_window_.height + 20;
    ShowText(show_image_, text_pt, str_buffer, g_red, 0.5, 1);

    date = PP_DATA(show_start_index_ + show_len_ - 1, PPI_trade_date);
    sprintf(str_buffer, "%04u-%02u-%02u", date / 10000, (date % 10000) / 100, date % 100);
    text_pt.x = vol_window_.x;
    text_pt.y = vol_window_.y + vol_window_.height + 20;
    ShowText(show_image_, text_pt, str_buffer, g_red, 0.5, 1);

    RefreshImageDynamic();
}

void ShowPPData::RefreshImageDynamic()
{
    show_image_.copyTo(show_image_dynamic_);
    if ((mouse_pt_.x > 0) || (mouse_pt_.y > 0))
    {
        Point pt1, pt2;
        pt1.x = 0;
        pt1.y = mouse_pt_.y;
        pt2.x = show_image_dynamic_.cols - 1;
        pt2.y = mouse_pt_.y;
        ShowLine(show_image_dynamic_, pt1, pt2, g_red);

        pt1.x = mouse_pt_.x;
        pt1.y = 0;
        pt2.x = mouse_pt_.x;
        pt2.y = show_image_dynamic_.rows - 1;
        ShowLine(show_image_dynamic_, pt1, pt2, g_red);
    }
    ShowActiveData();
}

void ShowPPData::ShowActiveData()
{
    if ((mouse_pt_.x >= price_window_.x) && (mouse_pt_.x <= (price_window_.x + price_window_.width)))
    {
        active_data_index_ = show_start_index_
                             + (double)(price_window_.x + price_window_.width - mouse_pt_.x) / price_window_.width * (show_len_ + 1)
                             - 0.5;
    }
    else
    {
        active_data_index_ = -1;
    }
    // printf("active_data_index_:%d\n", active_data_index_);
    if (active_data_index_ >= 0)
    {
        Point pt, text_pt;
        double active_value = PP_DATA(active_data_index_, PPI_close);
        char str_buffer[128];
        pt.x = price_window_.x + price_window_.width - (active_data_index_ - show_start_index_) * data_unit_width - data_unit_width;
        pt.y = price_window_.y + price_window_.height - (active_value - min_close) * close_ratio_;
        // printf("(%d, %d)\n", pt.x, pt.y);
        ShowPoint(show_image_dynamic_, pt, g_red);
        text_pt = pt;
        text_pt.x = text_pt.x < (image_size_.width - 200) ? (text_pt.x + 20) : (text_pt.x - 150);
        text_pt.y = text_pt.y > 40 ? text_pt.y - 20 : text_pt.y + 20;
        sprintf(str_buffer, "%.2f", active_value);
        ShowText(show_image_dynamic_, text_pt, str_buffer, g_red, 0.6, 2);

        active_value = PP_DATA(active_data_index_, PPI_vol);
        pt.y = vol_window_.y + vol_window_.height - active_value * vol_ratio_;
        ShowPoint(show_image_dynamic_, pt, g_red);
        text_pt = pt;
        text_pt.x = text_pt.x < (image_size_.width - 200) ? (text_pt.x + 20) : (text_pt.x - 150);
        text_pt.y = text_pt.y > 40 ? text_pt.y - 20 : text_pt.y + 20;
        sprintf(str_buffer, "%.0f", active_value);
        ShowText(show_image_dynamic_, text_pt, str_buffer, g_red, 0.6, 2);

        uint32_t date = PP_DATA(active_data_index_, PPI_trade_date);
        sprintf(str_buffer, "%04u-%02u-%02u", date / 10000, (date % 10000) / 100, date % 100);
        text_pt = pt;
        text_pt.x = text_pt.x < (image_size_.width - 200) ? (text_pt.x + 20) : (text_pt.x - 150);
        text_pt.y = vol_window_.y + vol_window_.height + 30;
        ShowText(show_image_dynamic_, text_pt, str_buffer, g_red, 0.6, 2);
    }
}

void ShowPPData::SetPPI(uint32_t ppi)
{
    if (ppi >= PPI_NUM)
    {
        return;
    }
    show_ppi_[ppi] = true;
}
