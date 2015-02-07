#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include <string>
#include <vector>
#include "mshadow/tensor.h"
#include <glog/logging.h>
#include "data.h"
#include "../utils/utils.h"
#include "../utils/io.h"
#include "iter_proc.hpp"
#include "iter_ball.hpp"
#include "iter_mult.hpp"
#include "iter_text.hpp"
#include "iter_char.hpp"
#include "iter_t_ball.hpp"
#include "iter_t_ball_label.hpp"
#include "iter_ball_label.hpp"
#include "iter_batch_text.hpp"

namespace layernet {
	IIterator<DataBatch>* CreateIterator(const DataProto& datap){
		IIterator<DataBatch>* it = NULL;
		if(datap.name() == "ball")
			it = new BALLIterator(datap);
		if(datap.name()=="t_ball")
			it=new T_BALLIterator(datap);
		if(datap.name()=="t_ball_label")
			it=new T_BALL_LABEL_Iterator(datap);
		if(datap.name()=="ball_label")
			it=new BALL_LABEL_Iterator(datap);
		if(datap.name()== "mult")
			it = new MULTIterator(datap);
		if(datap.name()== "text")
			it = new TEXTIterator(datap);
		if(datap.name()== "batch_text")
			it = new Batch_TEXTIterator(datap);
		if(datap.name()=="char")
			it = new CHARIterator(datap);
		CHECK(it != NULL)<<"You need to specify data";
		if(datap.thread()) it = new ThreadBufferIterator(it,datap);
		return it;
	}
}
;
