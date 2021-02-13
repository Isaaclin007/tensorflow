#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctime>
#include <math.h> 
#include <stdio.h>
#include "RBOEncrypt.h"
#include "VCAMath.h"

typedef struct
{
	unsigned int pos;
	unsigned int len;
}RBO_block_t;

unsigned int RBO_KeySupportDataLen(unsigned int *p_key_buffer, unsigned int key_len)
{
	unsigned int block_num=key_len/2;
	unsigned int max_data_end=0, temp_data_end;
	for(unsigned int iloop=0;iloop<block_num;iloop++)
	{
		temp_data_end=p_key_buffer[iloop*2]+p_key_buffer[iloop*2+1];
		if(max_data_end<temp_data_end)
		{
			max_data_end=temp_data_end;
		}
	}
	return max_data_end;
}

bool RBO_CreatKey(unsigned int max_data_len, unsigned int max_key_len, unsigned int *p_actual_key_len, unsigned int *p_key_buffer)
{
	if(max_key_len>RBO_MAX_KEY_LEN)
	{
		return false;
	}
	if(p_actual_key_len==NULL)
	{
		return false;
	}
	if(p_key_buffer==NULL)
	{
		return false;
	}
	unsigned int min_block_len, max_block_len, max_block_num, rand_num;

	max_block_num=(max_key_len/2);
	min_block_len=max_data_len/(max_block_num-1);
	max_block_len=min_block_len*2;

	CVcaRand vca_rand;
	RBO_block_t *RBO_block=(RBO_block_t *)malloc(sizeof(RBO_block_t)*max_block_num);
	unsigned int current_pos=0, block_num;
	unsigned int iloop;
	for(iloop=0;iloop<max_block_num;)
	{
		RBO_block[iloop].pos=current_pos;
		RBO_block[iloop].len=vca_rand.GetRand(min_block_len, max_block_len);
		current_pos+=RBO_block[iloop].len;
		iloop++;
		if(current_pos>=max_data_len)
		{
			break;
		}
	}
	block_num=iloop;
	*p_actual_key_len=block_num*2;

	RBO_block_t *p_key=(RBO_block_t *)p_key_buffer;
	for(iloop=0;iloop<block_num;iloop++)
	{
		rand_num=vca_rand.GetRand(0, block_num-1-iloop);
		p_key[iloop].pos=RBO_block[rand_num].pos;
		p_key[iloop].len=RBO_block[rand_num].len;

		RBO_block[rand_num].pos=RBO_block[block_num-1-iloop].pos;
		RBO_block[rand_num].len=RBO_block[block_num-1-iloop].len;
	}
	free(RBO_block);
	return true;
}


bool RBO_Encrypt(unsigned char *p_data, unsigned long data_len, unsigned char *p_encrypt_data, unsigned int *p_key_buffer, unsigned int key_len)
{
	if(p_key_buffer==NULL)
	{
		return false;
	}
	if(p_data==NULL)
	{
		return false;
	}
	if(p_encrypt_data==NULL)
	{
		return false;
	}
	if(key_len==0)
	{
		return false;
	}
	if((key_len%2)==1)
	{
		return false;
	}
	RBO_block_t *p_key=(RBO_block_t *)p_key_buffer;
	unsigned int current_pos=0, block_num;
	block_num=key_len/2;
	unsigned int copy_len;
	for(unsigned int iloop=0;iloop<block_num;iloop++)
	{
		if(p_key[iloop].pos<data_len)
		{
			if((p_key[iloop].pos+p_key[iloop].len)<=data_len)
			{
				copy_len=p_key[iloop].len;
			}
			else
			{
				copy_len=data_len-p_key[iloop].pos;
			}
			memcpy(p_encrypt_data+current_pos, p_data+p_key[iloop].pos, copy_len);
			current_pos+=copy_len;
		}
	}
	return true;
}


bool RBO_Decrypt(unsigned char *p_data, unsigned long data_len, unsigned char *p_decrypt_data, unsigned int *p_key_buffer, unsigned int key_len)
{
	if(p_key_buffer==NULL)
	{
		return false;
	}
	if(p_data==NULL)
	{
		return false;
	}
	if(p_decrypt_data==NULL)
	{
		return false;
	}
	if(key_len==0)
	{
		return false;
	}
	if((key_len%2)==1)
	{
		return false;
	}
	RBO_block_t *p_key=(RBO_block_t *)p_key_buffer;
	unsigned int current_pos=0, block_num;
	block_num=key_len/2;
	unsigned int copy_len;
	for(unsigned int iloop=0;iloop<block_num;iloop++)
	{
		if(p_key[iloop].pos<data_len)
		{
			if((p_key[iloop].pos+p_key[iloop].len)<=data_len)
			{
				copy_len=p_key[iloop].len;
			}
			else
			{
				copy_len=data_len-p_key[iloop].pos;
			}
			memcpy(p_decrypt_data+p_key[iloop].pos, p_data+current_pos, copy_len);
			current_pos+=copy_len;
		}
	}
	return true;
}


bool RBO_Encrypt(unsigned char *p_data, unsigned long data_len, unsigned int *p_key_buffer, unsigned int key_len)
{
	if(p_key_buffer==NULL)
	{
		return false;
	}
	if(p_data==NULL)
	{
		return false;
	}
	unsigned char *p_encrypt_data=(unsigned char *)malloc(data_len);
	if(p_encrypt_data==NULL)
	{
		return false;
	}
	RBO_Encrypt(p_data, data_len, p_encrypt_data, p_key_buffer, key_len);
	memcpy(p_data, p_encrypt_data, data_len);
	free(p_encrypt_data);
	return true;
}


bool RBO_Decrypt(unsigned char *p_data, unsigned long data_len, unsigned int *p_key_buffer, unsigned int key_len)
{
	if(p_key_buffer==NULL)
	{
		return false;
	}
	if(p_data==NULL)
	{
		return false;
	}
	unsigned char *p_decrypt_data=(unsigned char *)malloc(data_len);
	if(p_decrypt_data==NULL)
	{
		return false;
	}
	RBO_Decrypt(p_data, data_len, p_decrypt_data, p_key_buffer, key_len);
	memcpy(p_data, p_decrypt_data, data_len);
	free(p_decrypt_data);
	return true;
}




