//modifyed by bianjs, 20170320

#include <wchar.h>
#include <assert.h>
#include <locale.h>
#include <ctype.h>
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#else
#endif
#include "CvxText.h"

//====================================================================
//====================================================================

void CvxText::Init(const char *freeType)
{
   assert(freeType != NULL);

   // 打开字库文件, 创建一个字体

   if(FT_Init_FreeType(&m_library)) throw;
   if(FT_New_Face(m_library, freeType, 0, &m_face)) throw;

   // 设置字体输出参数

   restoreFont();

   // 设置C语言的字符集环境

   setlocale(LC_ALL, "");
}

CvxText::CvxText(void)
{
	
}


CvxText::CvxText(const char *freeType)
{
	Init(freeType);
}

// 释放FreeType资源

CvxText::~CvxText()
{
   FT_Done_Face    (m_face);
   FT_Done_FreeType(m_library);
}

// 设置字体参数:
//
// font         - 字体类型, 目前不支持
// size         - 字体大小/空白比例/间隔比例/旋转角度
// underline   - 下画线
// diaphaneity   - 透明度

void CvxText::getFont(int *type, CvScalar *size, bool *underline, float *diaphaneity)
{
   if(type) *type = m_fontType;
   if(size) *size = m_fontSize;
   if(underline) *underline = m_fontUnderline;
   if(diaphaneity) *diaphaneity = m_fontDiaphaneity;
}

void CvxText::setFont(int *type, CvScalar *size, bool *underline, float *diaphaneity)
{
   // 参数合法性检查

   if(type)
   {
      if(type >= 0) m_fontType = *type;
   }
   if(size)
   {
      m_fontSize.val[0] = fabs(size->val[0]);
      m_fontSize.val[1] = fabs(size->val[1]);
      m_fontSize.val[2] = fabs(size->val[2]);
      m_fontSize.val[3] = fabs(size->val[3]);
   }
   if(underline)
   {
      m_fontUnderline   = *underline;
   }
   if(diaphaneity)
   {
      m_fontDiaphaneity = *diaphaneity;
   }
   FT_Set_Pixel_Sizes(m_face, (int)m_fontSize.val[0], 0);
}

// 恢复原始的字体设置

void CvxText::restoreFont()
{
   m_fontType = 0;            // 字体类型(不支持)

   m_font_size=10;
   m_fontSize.val[0] = m_font_size;      // 字体大小
   m_fontSize.val[1] = 0.5;   // 空白字符大小比例
   m_fontSize.val[2] = 0.1;   // 间隔大小比例
   m_fontSize.val[3] = 0;      // 旋转角度(不支持)

   m_fontUnderline   = false;   // 下画线(不支持)

   m_fontDiaphaneity = 1.0;   // 色彩比例(可产生透明效果)

   // 设置字符大小

   FT_Set_Pixel_Sizes(m_face, (int)m_fontSize.val[0], 0);
   m_color=Scalar(255,255,255);
}

// 输出函数(颜色默认为黑色)

int CvxText::putText(IplImage *img, const char    *text, CvPoint pos)
{
   return putText(img, text, pos, CV_RGB(255,255,255));
}
int CvxText::putText(IplImage *img, const wchar_t *text, CvPoint pos)
{
   return putText(img, text, pos, CV_RGB(255,255,255));
}

//

int CvxText::putText(IplImage *img, const char    *text, CvPoint pos, CvScalar color)
{
   if(img == NULL) return -1;
   if(text == NULL) return -1;

   //

   int i;
   for(i = 0; text[i] != '\0'; ++i)
   {
      wchar_t wc = text[i];

      // 解析双字节符号

      if(!isascii(wc)) mbtowc(&wc, &text[i++], 2);

      // 输出当前的字符

      putWChar(img, wc, pos, color);
   }
   return i;
}
int CvxText::putText(IplImage *img, const wchar_t *text, CvPoint pos, CvScalar color)
{
   if(img == NULL) return -1;
   if(text == NULL) return -1;

   //

   int i;
   for(i = 0; text[i] != '\0'; ++i)
   {
      // 输出当前的字符

      putWChar(img, text[i], pos, color);
   }
   return i;
}

// 输出当前字符, 更新m_pos位置

void CvxText::putWChar(IplImage *img, wchar_t wc, CvPoint &pos, CvScalar color)
{
   // 根据unicode生成字体的二值位图

   FT_UInt glyph_index = FT_Get_Char_Index(m_face, wc);
   FT_Load_Glyph(m_face, glyph_index, 0);
   FT_Render_Glyph(m_face->glyph, FT_RENDER_MODE_NORMAL);

   //

   FT_GlyphSlot slot = m_face->glyph;

   // 行列数

   int rows = slot->bitmap.rows;
   int cols = slot->bitmap.width;
   int x_offset, y_offset;

   y_offset=pos.y+(m_face->size->metrics.ascender>>6) - slot->bitmap_top;
   x_offset=pos.x+slot->bitmap_left;
   if(x_offset>(img->width-(m_fontSize.val[0])))
   {
   	return;
   }
   if(y_offset>(img->height-(m_fontSize.val[0])))
   {
   	return;
   }

   //printf("\r\n%d,%d,%d,%d, (%d,%d), %d, %d, %u", cols, rows, slot->bitmap_left, slot->bitmap_top, slot->advance.x, slot->advance.y, m_face->size->metrics.ascender>>6,(m_face->size->metrics.max_advance>>6), wc);
   //

   int off =0;
   for(int i = 0; i < rows; ++i)
   {
      for(int j = 0; j < cols; ++j)
      {
	  	if(slot->bitmap.buffer[off]>0)
	  	{
			int r = y_offset+i;
			int c = x_offset+j;
			CvScalar scalar = cvGet2D(img, r, c);
			float p = m_fontDiaphaneity*(float)(slot->bitmap.buffer[off])/(float)255.0;
			for(int k = 0; k < 4; ++k)
			{
				scalar.val[k] = scalar.val[k]*(1-p) + color.val[k]*p;
			}
			cvSet2D(img, r, c, scalar);
	  	}
		off++;
#if 0   
2015-12-8 ,by bianjs
	printf("\r\nslot->bitmap.buffer[%d]=%02x", off, slot->bitmap.buffer[off]);
         if(slot->bitmap.buffer[off] & (0xC0 >> (j%8)))
         {
            int r = (img->origin==0)? pos.y - (rows-1-i): pos.y + i;;
            int c = pos.x + j;
         
            if(r >= 0 && r < img->height
               && c >= 0 && c < img->width)
            {
               CvScalar scalar = cvGet2D(img, r, c);

               // 进行色彩融合

               float p = m_fontDiaphaneity;
               for(int k = 0; k < 4; ++k)
               {
                  scalar.val[k] = scalar.val[k]*(1-p) + color.val[k]**p;
               }

               cvSet2D(img, r, c, scalar);
            }
         }
#endif
      } // end for
   } // end for

   // 修改下一个字的输出位置

   double space = m_fontSize.val[0]*m_fontSize.val[1];
   double sep   = m_fontSize.val[0]*m_fontSize.val[2];

   pos.x += (int)((rows? ((wc<128)? (cols+m_fontSize.val[0]*0.1):m_fontSize.val[0]): space) + sep);
}


void CvxText::setFont(int font_size, const Scalar &color, bool underline, float diaphaneity)
{	
	m_font_size=font_size;
	CvScalar text_size=Scalar(font_size,0.5,0.0,0);
	bool temp_underline=underline;
	float temp_diaphaneity=diaphaneity;
	setFont(NULL, &text_size, &temp_underline, &temp_diaphaneity);
	m_color=color;
}


void CvxText::putTextMat(Mat &input_image, Point &o_pt, char *p_msg_buffer)
{	
	IplImage temp_img=IplImage(input_image);
	int length=strlen(p_msg_buffer);
	int wlength;
#ifdef _WIN32
	wlength=MultiByteToWideChar(CP_ACP, 0, p_msg_buffer, length, m_caption, length+1);
#else
	wlength=mbstowcs(m_caption, p_msg_buffer, length*2);
#endif
	if(wlength<0)
	{
		return;
	}
	m_caption[wlength]=0;
	putText(&temp_img, m_caption, o_pt, m_color);
}

void CvxText::putTextMat(Mat &input_image, Point &o_pt, wchar_t *p_msg_buffer)
{	
	IplImage temp_img=IplImage(input_image);
	putText(&temp_img, p_msg_buffer, o_pt, m_color);
}


int CvxText::putTextMat(Mat &input_image, Rect &o_postion, char *p_msg_buffer)
{	
	int length=strlen(p_msg_buffer);
	int wlength;
#ifdef _WIN32
	wlength=MultiByteToWideChar(CP_ACP, 0, p_msg_buffer, length, m_caption, length+1);
#else
	wlength=mbstowcs(m_caption, p_msg_buffer, length*2);
#endif
	if(wlength<0)
	{
		return -1;
	}
	m_caption[wlength]=0;
	int line_num=0, row_num=0, max_row_num=(o_postion.width/m_font_size);
	double row_width=0;
	wchar_t *line_start_pointer=NULL;
	bool print_line=false;
	int y_area_offset=0;
	for(int iloop=0;iloop<(int)(wlength+1);iloop++)
	{
		if(line_start_pointer==NULL)
		{
			line_start_pointer=&m_caption[iloop];
		}
		if(m_caption[iloop]=='\n')
		{
			print_line=true;
		}
		else if(m_caption[iloop]==0)
		{
			if(row_num>0)
			{
				print_line=true;
			}
		}
		else
		{
			if(m_caption[iloop]<128)
			{
				row_width+=0.6;
			}
			else
			{
				row_width+=1.0;
			}
			row_num++;
			if(row_width>=max_row_num)
			{
				print_line=true;
			}
		}
		if(print_line)
		{
			memcpy(temp_buffer, line_start_pointer, sizeof(wchar_t)*row_num);
			temp_buffer[row_num]=0;
			Point temp_pt=Point(o_postion.x, o_postion.y+(int)(m_font_size*1.5*line_num));
			putTextMat(input_image, temp_pt, temp_buffer);
			line_start_pointer=NULL;
			line_num++;
			row_num=0;
			row_width=0;
			print_line=false;
			y_area_offset=(int)(m_font_size*1.5*line_num);
		}
	}
	return y_area_offset;
}


void CvxText::getTextSize(char *p_msg_buffer, Size &text_size)
{	
	Rect o_postion;
	o_postion.x=0;
	o_postion.y=0;
	o_postion.width=1000;
	o_postion.height=1000;
	wchar_t m_caption[1024];
	int length=strlen(p_msg_buffer);
	memset(m_caption, 0, sizeof(m_caption));
#ifdef _WIN32
	MultiByteToWideChar(CP_ACP, 0, p_msg_buffer, length, m_caption, length+1);
#else
	mbstowcs(m_caption, p_msg_buffer, length*2);
#endif
	int line_num=0, row_num=0, max_row_num=(o_postion.width/m_font_size);
	double row_width=0;
	wchar_t temp_buffer[256];
	wchar_t *line_start_pointer=NULL;
	bool print_line=false;
	int y_area_offset=0, x_area_offset=0, x_area_offset_max=0;
	for(int iloop=0;iloop<(int)(wcslen(m_caption)+1);iloop++)
	{
		if(line_start_pointer==NULL)
		{
			line_start_pointer=&m_caption[iloop];
		}
		if(m_caption[iloop]=='\n')
		{
			print_line=true;
		}
		else if(m_caption[iloop]==0)
		{
			print_line=true;
		}
		else
		{
			if(m_caption[iloop]<128)
			{
				row_width+=0.5;
			}
			else
			{
				row_width+=1.0;
			}
			row_num++;
			if(row_width>=max_row_num)
			{
				print_line=true;
			}
		}
		if(print_line)
		{
			memcpy(temp_buffer, line_start_pointer, sizeof(wchar_t)*row_num);
			temp_buffer[row_num]=0;
			Point temp_pt=Point(o_postion.x, o_postion.y+(int)(m_font_size*1.5*line_num));
			//putTextMat(input_image, temp_pt, temp_buffer);
			line_start_pointer=NULL;
			line_num++;
			row_num=0;
			row_width=0;
			print_line=false;
			x_area_offset=row_width*m_font_size;
			if(x_area_offset_max<x_area_offset)
			{
				x_area_offset_max=x_area_offset;
			}
			y_area_offset=(int)(m_font_size*1.5*line_num);
		}
	}
	text_size.width=x_area_offset_max;
	text_size.height=y_area_offset;
}



