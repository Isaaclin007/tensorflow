#ifndef _VCA_ACC_H_
#define _VCA_ACC_H_

#include "VCALogManage.h"
#include "VCAImage.h"

#define ACC_MAX_OBJECTS_NUM 256
#define ACC_MAX_CLASS_NUM 1000
#define ACC_MAX_SIZE_NUM 100

typedef struct
{
	unsigned int objects_num;
	deepnet_object_t objects[ACC_MAX_OBJECTS_NUM];
}acc_objects_t;

typedef struct
{
	unsigned int currect_num;
	unsigned int missed_num;
	unsigned int error_num;
}acc_statistics_unit_t;

typedef struct
{
	acc_statistics_unit_t classes[ACC_MAX_CLASS_NUM];
	acc_statistics_unit_t sizes[ACC_MAX_SIZE_NUM];
	acc_statistics_unit_t sum;
}acc_statistics_confidence_t;

typedef struct
{
	acc_statistics_confidence_t confidence[101];
}acc_statistics_t;

typedef struct
{
	int size_num;
	double size_step;
	double sizes[ACC_MAX_SIZE_NUM];
}acc_config_t;

class  CVCAObjectDetectAcc
{
public:
	CVCAObjectDetectAcc(void);
	bool Clear(void);
	double IoU(deepnet_object_t *p_object_a, deepnet_object_t *p_object_b);
	bool InputData(acc_objects_t *p_detect_objects, acc_objects_t *p_tag_objects, acc_statistics_confidence_t *p_statistics);
	bool InputData(acc_objects_t *p_detect_objects, acc_objects_t *p_tag_objects);
	bool SetSize(double size_step);
	unsigned int SizeIndex(Rect &o_object);
	bool Empty(acc_statistics_unit_t *p_statistics_unit);
	double Acc(acc_statistics_unit_t *p_statistics_unit);
	unsigned int MaxAccConfidence(void);
	bool OutputStatisticsUnit(acc_statistics_unit_t *p_statistics_unit, char *p_caption, CVCALogManage &o_log);
	bool OutputStatistics(const char *p_filename, acc_statistics_confidence_t *p_statistics, unsigned int confidence);
	bool OutputStatistics(const char *p_filename);

	acc_statistics_t m_statistics;
	acc_config_t m_config;
	unsigned int m_input_count;
};

#endif


