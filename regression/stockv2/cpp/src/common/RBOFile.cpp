#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctime>
#include <math.h> 
#include <stdio.h>
#include "RBOEncrypt.h"
#include "DESLib.h"
#include "VCAFileAccess.h"
#include "RBOFile.h"
#include "VCALogManage.h"

CRBOFile::CRBOFile(void)
{
	m_p_data=NULL;
	m_data_len=0;
}

bool CRBOFile::Load(const char *p_file_name)
{
	Free();
	m_data_len=VcaFileSize(p_file_name);
	if(m_data_len==0)
	{
		LOG_ERROR;
		return false;
	}
	m_p_data=(unsigned char *)malloc(m_data_len);
	if(m_p_data==NULL)
	{
		LOG_ERROR;
		return false;
	}
	if(!LoadBinFile(p_file_name, m_p_data, m_data_len))
	{
		LOG_ERROR;
		return false;
	}
#if 0
	if(m_data_len<=RBO_KeySupportDataLen(RBO_key, RBO_key_len))
	{
		if(!RBO_Decrypt(m_p_data, m_data_len, RBO_key, RBO_key_len))
		{
			LOG_ERROR;
			return false;
		}
	}
	else
	{
		if(!RBO_Decrypt(m_p_data, m_data_len, RBO_key_1000M, RBO_key_len_1000M))
		{
			LOG_ERROR;
			return false;
		}
	}
#endif
	if(m_data_len<1000000)
	{
		unsigned char temp_output_buffer[1000000];
		unsigned long temp_output_len;
		if(!DES_Decrypt_data(g_des_key, m_p_data, m_data_len, temp_output_buffer, &temp_output_len))
		{
			LOG_ERROR;
			return false;
		}
		memcpy(m_p_data, temp_output_buffer, m_data_len);
	}
	else
	{
		if(!RBO_Decrypt(m_p_data, m_data_len, RBO_key_1000M, RBO_key_len_1000M))
		{
			LOG_ERROR;
			return false;
		}
	}
	return true;
}

bool CRBOFile::Save(char *p_file_name)
{
	if(m_data_len==0)
	{
		LOG_ERROR;
		return false;
	}
	if(m_p_data==NULL)
	{
		LOG_ERROR;
		return false;
	}
	return SaveBinData(p_file_name, m_p_data, m_data_len);
}

bool CRBOFile::Free(void)
{
	if(m_p_data)
	{
		free(m_p_data);
		m_p_data=NULL;
	}
	return true;
}



