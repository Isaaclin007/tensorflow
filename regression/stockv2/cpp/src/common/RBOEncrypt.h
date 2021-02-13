#pragma once

//reorder block 

#define RBO_MAX_KEY_LEN 200000

unsigned int RBO_KeySupportDataLen(unsigned int *p_key_buffer, unsigned int key_len);
bool RBO_CreatKey(unsigned int max_data_len, unsigned int max_key_len, unsigned int *p_actual_key_len, unsigned int *p_key_buffer);
bool RBO_Encrypt(unsigned char *p_data, unsigned long data_len, unsigned char *p_encrypt_data, unsigned int *p_key_buffer, unsigned int key_len);
bool RBO_Decrypt(unsigned char *p_data, unsigned long data_len, unsigned char *p_decrypt_data, unsigned int *p_key_buffer, unsigned int key_len);
bool RBO_Encrypt(unsigned char *p_data, unsigned long data_len, unsigned int *p_key_buffer, unsigned int key_len);
bool RBO_Decrypt(unsigned char *p_data, unsigned long data_len, unsigned int *p_key_buffer, unsigned int key_len);

extern unsigned int RBO_key_len;
extern unsigned int RBO_key[];
extern unsigned int RBO_key_len_1000M;
extern unsigned int RBO_key_1000M[];


