#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "xm_acc.h"
#include "VCALogManage.h"


CVCAObjectDetectAcc::CVCAObjectDetectAcc(void)
{
	Clear();
	SetSize(1.5);
}

bool CVCAObjectDetectAcc::Clear(void)
{
	memset(&m_statistics, 0, sizeof(acc_statistics_t));
	m_input_count=0;
	return true;
}

double CVCAObjectDetectAcc::IoU(deepnet_object_t *p_object_a, deepnet_object_t *p_object_b)
{
	if(p_object_a->class_id==p_object_b->class_id)
	{
		return ObjectsCrossIoU(p_object_a->rect, p_object_b->rect);
	}
	else
	{
		return 0.0;
	}
}

bool CVCAObjectDetectAcc::InputData(acc_objects_t *p_detect_objects, acc_objects_t *p_tag_objects, acc_statistics_confidence_t *p_statistics)
{
	unsigned int tloop, dloop;
	unsigned char detect_objects_match[ACC_MAX_OBJECTS_NUM], tag_objects_match[ACC_MAX_OBJECTS_NUM];
	
	for(dloop=0;dloop<p_detect_objects->objects_num;dloop++)
	{
		detect_objects_match[dloop]=0;
		if(p_detect_objects->objects[dloop].class_id>=ACC_MAX_CLASS_NUM)
		{
			LOG_ERROR;
			return false;
		}
	}
	for(tloop=0;tloop<p_tag_objects->objects_num;tloop++)
	{
		tag_objects_match[tloop]=0;
		if(p_tag_objects->objects[dloop].class_id>=ACC_MAX_CLASS_NUM)
		{
			LOG_ERROR;
			return false;
		}
	}

	for(dloop=0;dloop<p_detect_objects->objects_num;dloop++)
	{
		for(tloop=0;tloop<p_tag_objects->objects_num;tloop++)
		{
			if(IoU(&p_detect_objects->objects[dloop], &p_tag_objects->objects[tloop])>=0.5)
			{
				detect_objects_match[dloop]=1;
				tag_objects_match[tloop]=1;
			}
		}
	}
	for(dloop=0;dloop<p_detect_objects->objects_num;dloop++)
	{
		if(detect_objects_match[dloop]==1)
		{
			
		}
		else
		{
			p_statistics->classes[p_detect_objects->objects[dloop].class_id].error_num++;
			p_statistics->sizes[SizeIndex(p_detect_objects->objects[dloop].rect)].error_num++;
			p_statistics->sum.error_num++;
		}
	}
	for(tloop=0;tloop<p_tag_objects->objects_num;tloop++)
	{
		if(tag_objects_match[tloop]==1)
		{
			p_statistics->classes[p_tag_objects->objects[tloop].class_id].currect_num++;
			p_statistics->sizes[SizeIndex(p_tag_objects->objects[tloop].rect)].currect_num++;
			p_statistics->sum.currect_num++;
		}
		else
		{
			p_statistics->classes[p_tag_objects->objects[tloop].class_id].missed_num++;
			p_statistics->sizes[SizeIndex(p_tag_objects->objects[tloop].rect)].missed_num++;
			p_statistics->sum.missed_num++;
		}
	}
	return true;
}

bool CVCAObjectDetectAcc::InputData(acc_objects_t *p_detect_objects, acc_objects_t *p_tag_objects)
{
	acc_objects_t temp_objects;

	for(unsigned int cloop=0;cloop<=100;cloop++)
	{
		memset(&temp_objects, 0, sizeof(acc_objects_t));
		for(unsigned int iloop=0;iloop<p_detect_objects->objects_num;iloop++)
		{
			if(p_detect_objects->objects[iloop].score>=cloop)
			{
				memcpy(&temp_objects.objects[temp_objects.objects_num], &p_detect_objects->objects[iloop], sizeof(deepnet_object_t));
				temp_objects.objects_num++;
			}
		}
		InputData(&temp_objects, p_tag_objects, &m_statistics.confidence[cloop]);
	}
	m_input_count++;
	return true;
}


bool CVCAObjectDetectAcc::SetSize(double size_step)
{
	if(size_step<1.0)
	{
		LOG_ERROR;
		return false;
	}
	m_config.size_step=size_step;
	m_config.sizes[0]=10.0;
	m_config.size_num=1;
	for(int iloop=1;iloop<ACC_MAX_SIZE_NUM;iloop++)
	{
		m_config.sizes[iloop]=m_config.sizes[iloop-1]*m_config.size_step;
		m_config.size_num++;
		if(m_config.sizes[iloop]>4000000)
		{
			break;
		}
	}
	m_config.sizes[0]=0.0;
	return true;
}

unsigned int CVCAObjectDetectAcc::SizeIndex(Rect &o_object)
{
	double temp_size=o_object.width*o_object.height;
	for(int iloop=0;iloop<m_config.size_num-1;iloop++)
	{
		if((temp_size>=m_config.sizes[iloop])&&(temp_size<m_config.sizes[iloop+1]))
		{
			return iloop;
		}
	}
	return (m_config.size_num-1);
}

bool CVCAObjectDetectAcc::Empty(acc_statistics_unit_t *p_statistics_unit)
{
	return ((p_statistics_unit->currect_num==0)&&(p_statistics_unit->missed_num==0)&&(p_statistics_unit->error_num==0));
}


double CVCAObjectDetectAcc::Acc(acc_statistics_unit_t *p_statistics_unit)
{
	if(p_statistics_unit->currect_num==0)
	{
		return 0.0;
	}
	return ((double)p_statistics_unit->currect_num)/((double)(p_statistics_unit->currect_num+p_statistics_unit->missed_num+p_statistics_unit->error_num));
}

bool CVCAObjectDetectAcc::OutputStatisticsUnit(acc_statistics_unit_t *p_statistics_unit, char *p_caption, CVCALogManage &o_log)
{
	if(!Empty(p_statistics_unit))
	{
		o_log.print_without_time("%-16s%-10.4f%-10u%-10u%-10u%-10u", 
			p_caption, 
			Acc(p_statistics_unit), 
			p_statistics_unit->currect_num, 
			p_statistics_unit->missed_num, 
			p_statistics_unit->error_num, 
			p_statistics_unit->currect_num+p_statistics_unit->missed_num+p_statistics_unit->error_num
			);
	}
	return true;
}

unsigned int CVCAObjectDetectAcc::MaxAccConfidence(void)
{
	double max_acc=0.0, temp_acc;
	unsigned int max_acc_index=0;

	for(int cloop=0;cloop<=100;cloop++)
	{
		temp_acc=Acc(&m_statistics.confidence[cloop].sum);
		if(temp_acc>max_acc)
		{
			max_acc=temp_acc;
			max_acc_index=cloop;
		}
	}
	return max_acc_index;
}

bool CVCAObjectDetectAcc::OutputStatistics(const char *p_filename, acc_statistics_confidence_t *p_statistics, unsigned int confidence)
{
	CVCALogManage temp_log(p_filename);
	char temp_caption[256];

	temp_log.print_without_time("confidence: %u", confidence);
	temp_log.print_without_time("\n");
	
	temp_log.print_without_time("%-16s%-10s%-10s%-10s%-10s%-10s", "class", "acc", "currect", "missed", "error", "total");
	temp_log.print_without_time("------------------------------------------------------------");
	sprintf(temp_caption, "sum");
	OutputStatisticsUnit(&p_statistics->sum, temp_caption, temp_log);
	for(int iloop=0;iloop<ACC_MAX_CLASS_NUM;iloop++)
	{
		sprintf(temp_caption, "%d", iloop);
		OutputStatisticsUnit(&p_statistics->classes[iloop], temp_caption, temp_log);
	}
	

	temp_log.print_without_time("\n");
	temp_log.print_without_time("%-16s%-10s%-10s%-10s%-10s%-10s", "size", "acc", "currect", "missed", "error", "total");
	temp_log.print_without_time("------------------------------------------------------------");
	for(int iloop=0;iloop<m_config.size_num-1;iloop++)
	{
		sprintf(temp_caption, "%.0f-%.0f", m_config.sizes[iloop], m_config.sizes[iloop+1]);
		OutputStatisticsUnit(&p_statistics->sizes[iloop], temp_caption, temp_log);
	}
	sprintf(temp_caption, "%.0f-", m_config.sizes[m_config.size_num-1]);
	OutputStatisticsUnit(&p_statistics->sizes[m_config.size_num-1], temp_caption, temp_log);
	return true;
}


bool CVCAObjectDetectAcc::OutputStatistics(const char *p_filename)
{
	if(m_input_count>0)
	{
		unsigned int temp_confidence=MaxAccConfidence();
		return OutputStatistics(p_filename, &m_statistics.confidence[temp_confidence], temp_confidence);
	}
	return true;
}



