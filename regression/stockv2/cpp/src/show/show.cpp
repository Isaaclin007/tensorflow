
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <gflags/gflags.h>
#include "show_pp_data.h"
#include "VCASearchFiles.h"

using namespace std;


DEFINE_bool(r, false, "rand show");
DEFINE_string(c, "000001.SZ", "ts code");
DEFINE_int32(l, 100, "rand test show data len");

#ifdef USE_OPENCV
void showPathPlan(Path_t nodes, Path_t path) {
    Mat img(600, 1000, CV_8UC3, Scalar(0, 0, 0));
    for (int iloop = 0; iloop < nodes.size() -1; iloop++) {
        line(img, Point(nodes[iloop].x, nodes[iloop].y),
             Point(nodes[iloop + 1].x, nodes[iloop + 1].y), Scalar(255, 255, 0), 2, CV_AA);
    }
    for (int iloop = 0; iloop < nodes.size(); iloop++) {
        circle(img, Point(nodes[iloop].x, nodes[iloop].y), 5, Scalar(0, 255, 0), -1, CV_AA);
    }
    for (int iloop = 0; iloop < path.size(); iloop++) {
        circle(img, Point(path[iloop].x, path[iloop].y), 5, Scalar(0, 0, 255), -1, CV_AA);
    }
    imshow("path_plan", img);
    waitKey(-1);
}
#endif

vector<char *> v_file_name;

void SearchFileCallback(void *p_cbk_data, const char* cbk_root_path_name, const char *cbk_sub_path_name, const char *cbk_file_name)
{
    char * file_name = (char *)malloc(1024);
    sprintf(file_name, "%s%s/%s", cbk_root_path_name, cbk_sub_path_name, cbk_file_name);
    v_file_name.push_back(file_name);
}

int main(int argc, char *argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    ShowPPData show;

    SearchFiles("../data/preprocessed", "", NULL, SearchFileCallback);

    if (FLAGS_r)
    {
        while (1)
        {
            struct NpyArray a = npy_load(v_file_name[rand() % v_file_name.size()]);
            double *pp_data = a.data<double>();
            if (a.shape[0] <= 200)
            {
                continue;
            }
            show.SetData(pp_data, a.shape[0]);
            show.SetShowData(rand() % (a.shape[0] - FLAGS_l * 2) + FLAGS_l * 2, FLAGS_l);
            show.Run();
        }
    }
    else
    {
        for (int iloop = 0; iloop < v_file_name.size(); iloop++)
        {
            if (strcasestr(v_file_name[iloop], FLAGS_c.c_str()))
            {
                struct NpyArray a = npy_load(v_file_name[iloop]);
                double *pp_data = a.data<double>();
                show.SetData(pp_data, a.shape[0]);
                show.SetShowData(0, a.shape[0]);
                show.Run();
            }
        }
    }
    return 0;
}
