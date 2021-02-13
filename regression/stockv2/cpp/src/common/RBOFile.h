#pragma once



class CRBOFile
{
public:
	CRBOFile(void);
	bool Load(const char *p_file_name);
	bool Save(char *p_file_name);
	bool Free(void);

	unsigned char *m_p_data;
	unsigned long m_data_len;
};


