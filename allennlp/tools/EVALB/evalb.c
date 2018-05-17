/*****************************************************************/
/* evalb [-p param_file] [-dh] [-e n] gold-file test-file        */
/*                                                               */
/*        Evaluate bracketing in test-file against gold-file.    */
/*        Return recall, precision, tagging accuracy.            */
/*                                                               */
/*   <option>                                                    */
/*        -p param_file  parameter file                          */
/*        -d             debug mode                              */
/*        -e n           number of error to kill (default=10)    */
/*        -h             help                                    */
/*                                                               */
/*                                         Satoshi Sekine (NYU)  */
/*                                         Mike Collins (UPenn)  */
/*                                                               */
/*                                         October.1997          */
/*                                                               */
/* Please refer README for the update information                */
/*****************************************************************/

#include <stdio.h>
#include <stdlib.h> //### added for exit, atoi decls
#include <ctype.h>
#include <string.h>



/* Internal Data format -------------------------------------------*/
/*                                                                 */
/* (S (NP (NNX this)) (VP (VBX is) (NP (DT a) (NNX pen))) (SYM .)) */
/*                                                                 */
/*   wn=5                                                          */
/*                        word    label                            */
/*   terminal[0] =        this     NNX                             */
/*   terminal[1] =        is       VBX                             */
/*   terminal[2] =        a        DT                              */
/*   terminal[3] =        pen      NNX                             */
/*   terminal[4] =        .        SYM                             */
/*                                                                 */
/*   bn=4                                                          */
/*                      start     end      label                   */
/*   bracket[0]  =        0        5         S                     */
/*   bracket[1]  =        0        0         NP                    */
/*   bracket[2]  =        1        4         VP                    */
/*   bracket[3]  =        2        4         NP                    */
/*                                                                 */
/*              matched bracketing                                 */
/*   Recall = ---------------------------                          */
/*             # of bracket in ref-data                            */
/*                                                                 */
/*              matched bracketing                                 */
/*   Recall = ---------------------------                          */
/*             # of bracket in test-data                           */
/*                                                                 */
/*-----------------------------------------------------------------*/

/******************/
/* constant macro */
/******************/

#define MAX_SENT_LEN           5000
#define MAX_WORD_IN_SENT        200
#define MAX_BRACKET_IN_SENT     200
#define MAX_WORD_LEN            100
#define MAX_LABEL_LEN            30
#define MAX_QUOTE_TERM           20

#define MAX_DELETE_LABEL        100
#define MAX_EQ_LABEL            100
#define MAX_EQ_WORD             100

#define MAX_LINE_LEN            500

#define DEFAULT_MAX_ERROR        10
#define DEFAULT_CUT_LEN          40

/*************/
/* structure */
/*************/

typedef struct ss_terminal {
    char word[MAX_WORD_LEN];
    char label[MAX_LABEL_LEN];
    int  result;                /* 0:unmatch, 1:match, 9:undef */
} s_terminal;

typedef struct ss_term_ind {
	s_terminal term;
	int index;
   int bracket;
   int endslen;
   int ends[MAX_BRACKET_IN_SENT];
} s_term_ind;

typedef struct ss_bracket {
    int start;
    int end;
    unsigned int buf_start;
    unsigned int buf_end;
    char label[MAX_LABEL_LEN];
    int  result;                 /* 0: unmatch, 1:match, 5:delete 9:undef */
} s_bracket;


typedef struct ss_equiv {
    char *s1;
    char *s2;
} s_equiv;


/****************************/
/* global variables         */
/*   gold-data: suffix = 1  */
/*   test-data: suffix = 2  */
/****************************/

/*---------------*/
/* Sentence data */
/*---------------*/
int wn1, wn2;                              /* number of words in sentence  */
int r_wn1;                                 /* number of words in sentence  */
                                           /* which only ignores labels in */
                                           /* DELETE_LABEL_FOR_LENGTH      */

s_terminal terminal1[MAX_WORD_IN_SENT];    /* terminal information */
s_terminal terminal2[MAX_WORD_IN_SENT];

s_term_ind quotterm1[MAX_QUOTE_TERM];      /* special terminals ("'","POS") */
s_term_ind quotterm2[MAX_QUOTE_TERM];

int bn1, bn2;                              /* number of brackets */

int r_bn1, r_bn2;                          /* number of brackets */
                                           /* after deletion */

s_bracket bracket1[MAX_BRACKET_IN_SENT];   /* bracket information */
s_bracket bracket2[MAX_BRACKET_IN_SENT];


/*------------*/
/* Total data */
/*------------*/
int TOTAL_bn1, TOTAL_bn2, TOTAL_match;     /* total number of brackets */
int TOTAL_sent;                            /* No. of sentence */
int TOTAL_error_sent;                      /* No. of error sentence */
int TOTAL_skip_sent;                       /* No. of skip sentence */
int TOTAL_comp_sent;                       /* No. of complete match sent */
int TOTAL_word;                            /* total number of word */
int TOTAL_crossing;                        /* total crossing */
int TOTAL_no_crossing;                     /* no crossing sentence */
int TOTAL_2L_crossing;                     /* 2 or less crossing sentence */
int TOTAL_correct_tag;                     /* total correct tagging */

int TOT_cut_len = DEFAULT_CUT_LEN;         /* Cut-off length in statistics */

                                 /* data for sentences with len <= CUT_LEN */
                                 /* Historically it was 40.                */
int TOT40_bn1, TOT40_bn2, TOT40_match;     /* total number of brackets */
int TOT40_sent;                            /* No. of sentence */
int TOT40_error_sent;                      /* No. of error sentence */
int TOT40_skip_sent;                       /* No. of skip sentence */
int TOT40_comp_sent;                       /* No. of complete match sent */
int TOT40_word;                            /* total number of word */
int TOT40_crossing;                        /* total crossing */
int TOT40_no_crossing;                     /* no crossing sentence */
int TOT40_2L_crossing;                     /* 2 or less crossing sentence */
int TOT40_correct_tag;                     /* total correct tagging */

/*------------*/
/* miscallous */
/*------------*/
int Line;                                  /* line number */
int Error_count = 0;                       /* Error count */
int Status;                                /* Result status for each sent */
                                           /*    0: OK, 1: skip, 2: error */

/*-------------------*/
/* stack manuplation */
/*-------------------*/
int stack_top;
int stack[MAX_BRACKET_IN_SENT];

/************************************************************/
/* User parameters which can be specified in parameter file */
/************************************************************/

/*------------------------------------------*/
/* Debug mode                               */
/*   print out data for individual sentence */
/*------------------------------------------*/
int DEBUG=0;

/*------------------------------------------*/
/* MAX error                                */
/*    Number of error to stop the process.  */
/*    This is useful if there could be      */
/*    tokanization error.                   */
/*    The process will stop when this number*/
/*    of errors are accumulated.            */
/*------------------------------------------*/
int Max_error = DEFAULT_MAX_ERROR;

/*------------------------------------------*/
/* Cut-off length for statistics            */
/*    int TOT_cut_len = DEFAULT_CUT_LEN;    */
/*    (Defined above)                       */
/*------------------------------------------*/


/*------------------------------------------*/
/* unlabeled or labeled bracketing          */
/*    0: unlabeled bracketing               */
/*    1: labeled bracketing                 */
/*------------------------------------------*/
int F_label    = 1;                 

/*------------------------------------------*/
/* Delete labels                            */
/*    list of labels to be ignored.         */
/*    If it is a pre-terminal label, delete */
/*    the word along with the brackets.     */
/*    If it is a non-terminal label, just   */
/*    delete the brackets (don't delete     */
/*    childrens).                           */
/*------------------------------------------*/
char *Delete_label[MAX_DELETE_LABEL];
int Delete_label_n = 0;

/*------------------------------------------*/
/* Delete labels for length calculation     */
/*    list of labels to be ignored for      */
/*    length calculation purpose            */
/*------------------------------------------*/
char *Delete_label_for_length[MAX_DELETE_LABEL];
int Delete_label_for_length_n = 0;

/*------------------------------------------*/
/* Labels to be considered for misquote     */
/*    (could be possesive or quote)         */
/*------------------------------------------*/
char *Quote_term[MAX_QUOTE_TERM];
int Quote_term_n = 0;

/*------------------------------------------*/
/* Equivalent labels, words                 */
/*     the pairs are considered equivalent  */
/*     This is non-directional.             */
/*------------------------------------------*/
s_equiv EQ_label[MAX_EQ_LABEL];
int EQ_label_n = 0;

s_equiv EQ_word[MAX_EQ_WORD];
int EQ_word_n = 0;



/************************/
/* Function return-type */
/************************/
int main();
void init_global();
void print_head();
void init();
void read_parameter_file();
void set_param();
int narg();
int read_line();

void pushb();
int popb();
int stackempty();

void calc_result(unsigned char *buf1,unsigned char *buf);
void fix_quote();
void reinsert_term();
void massage_data();
void modify_label();
void individual_result();
void print_total();
void dsp_info();
int is_terminator();
int is_deletelabel();
int is_deletelabel_for_length();
int is_quote_term();
int word_comp();
int label_comp();

void Error();
void Fatal();
void Usage();

/* ### provided by std headers 
int fprintf();
int printf();
int atoi();
int fclose();
int sscanf();
*/

/***********/
/* program */
/***********/
#define ARG_CHECK(st) if(!(*++(*argv) || (--argc && *++argv))){ \
			 fprintf(stderr,"Missing argument: %s\n",st); \
		      }

int
main(argc,argv)
int argc;
char *argv[];
{
    char *filename1, *filename2;
    FILE *fd1, *fd2;
    unsigned char buff[5000];
    unsigned char buff1[5000];

    filename1=NULL;
    filename2=NULL;

    for(argc--,argv++;argc>0;argc--,argv++){
	if(**argv == '-'){
	    while(*++(*argv)){
		switch(**argv){

		  case 'h':    /* help */
		    Usage();
		    exit(1);

		  case 'd':      /* debug mode */
		    DEBUG = 1;
		    goto nextarg;

		  case 'D':      /* debug mode */
		    DEBUG = 2;
		    goto nextarg;

		  case 'c':      /* cut-off length */
		    ARG_CHECK("cut-off length for statistices");
		    TOT_cut_len = atoi(*argv);
		    goto nextarg;

		  case 'e':      /* max error */
		    ARG_CHECK("number of error to kill");
		    Max_error = atoi(*argv);
		    goto nextarg;

		  case 'p':      /* parameter file */
		    ARG_CHECK("parameter file");
		    read_parameter_file(*argv);
		    goto nextarg;

		  default:
		    Usage();
		    exit(0);
		}
	    }
	} else {
	    if(filename1==NULL){
		filename1 = *argv;
	    }else if(filename2==NULL){
		filename2 = *argv;
	    }
	}
      nextarg: continue;
    }

    init_global();


    if((fd1 = fopen(filename1,"r"))==NULL){
	Fatal("Can't open gold file (%s)\n",filename1);
    }
    if((fd2 = fopen(filename2,"r"))==NULL){
	Fatal("Can't open test file (%s)\n",filename2);
    }

    print_head();

    for(Line=1;fgets(buff,5000,fd1)!=NULL;Line++){
    
	init();

      /* READ 1 */
	r_wn1 = read_line(buff,terminal1,quotterm1,&wn1,bracket1,&bn1);

	strcpy(buff1,buff);

      /* READ 2 */
	if(fgets(buff,5000,fd2)==NULL){
	    Error("Number of lines unmatch (too many lines in gold file)\n");
	    break;
	}

	read_line(buff,terminal2,quotterm2,&wn2,bracket2,&bn2);

      /* Calculate result and print it */
	calc_result(buff1,buff);

	if(DEBUG>=1){
	    dsp_info();
	}
    }

    if(fgets(buff,5000,fd2)!=NULL){
	Error("Number of lines unmatch (too many lines in test file)\n");
    }

    print_total();

    return (0);
}


/*-----------------------------*/
/* initialize global variables */
/*-----------------------------*/
void
init_global()
{
    TOTAL_bn1 = TOTAL_bn2 = TOTAL_match = 0;
    TOTAL_sent = TOTAL_error_sent = TOTAL_skip_sent = TOTAL_comp_sent = 0;
    TOTAL_word = TOTAL_correct_tag = 0;
    TOTAL_crossing = 0;
    TOTAL_no_crossing = TOTAL_2L_crossing = 0;

    TOT40_bn1 = TOT40_bn2 = TOT40_match = 0;
    TOT40_sent = TOT40_error_sent = TOT40_skip_sent = TOT40_comp_sent = 0;
    TOT40_word = TOT40_correct_tag = 0;
    TOT40_crossing = 0;
    TOT40_no_crossing = TOT40_2L_crossing = 0;

}


/*------------------*/
/* print head title */
/*------------------*/
void
print_head()
{
    printf("  Sent.                        Matched  Bracket   Cross        Correct Tag\n");
    printf(" ID  Len.  Stat. Recal  Prec.  Bracket gold test Bracket Words  Tags Accracy\n");
    printf("============================================================================\n");
}


/*-----------------------------------------------*/
/* initialization at each individual computation */
/*-----------------------------------------------*/
void
init()
{
  int i;

  wn1 = 0;
  wn2 = 0;
  bn1 = 0;
  bn2 = 0;
  r_bn1 = 0;
  r_bn2 = 0;

  for(i=0;i<MAX_WORD_IN_SENT;i++){
      terminal1[i].word[0]  = '\0';
      terminal1[i].label[0] = '\0';
      terminal1[i].result   = 9;
      terminal2[i].word[0]  = '\0';
      terminal2[i].label[0] = '\0';
      terminal2[i].result   = 9;
  }

  for(i=0;i<MAX_QUOTE_TERM;i++){
      quotterm1[i].term.word[0]  = '\0';
      quotterm1[i].term.label[0] = '\0';
      quotterm1[i].term.result   = 9;
      quotterm1[i].index         = -1;
      quotterm1[i].bracket       = -1;
      quotterm2[i].term.word[0]  = '\0';
      quotterm2[i].term.label[0] = '\0';
      quotterm2[i].term.result   = 9;
      quotterm2[i].index         = -1;
      quotterm2[i].bracket       = -1;
  }

  for(i=0;i<MAX_BRACKET_IN_SENT;i++){
      bracket1[i].start    = -1;
      bracket1[i].end      = -1;
      bracket1[i].label[0] = '\0';
      bracket1[i].result   = 9;
      bracket2[i].start    = -1;
      bracket2[i].end      = -1;
      bracket2[i].label[0] = '\0';
      bracket2[i].result   = 9;
  }

  Status = 0;
}

/*----------------*/
/* parameter file */
/*----------------*/
void
read_parameter_file(filename)
char *filename;
{
    char buff[MAX_LINE_LEN];
    FILE *fd;
    int line;
    int i;

    if((fd=fopen(filename,"r"))==NULL){
	Fatal("Can't open parameter file (%s)\n",filename);
    }

    for(line=1;fgets(buff,MAX_LINE_LEN,fd)!=NULL;line++){

      /* clean up the tail and find unvalid line */
      /*-----------------------------------------*/
	for(i=strlen(buff)-1;i>0 && (isspace(buff[i]) || buff[i]=='\n');i--){
	    buff[i]='\0';
	}
	if(buff[0]=='#' ||      /* comment-line */
	   strlen(buff)<3){     /* too short, just ignore */
	    continue;
	}

      /* place the parameter and value */
      /*-------------------------------*/
	for(i=0;!isspace(buff[i]);i++);
	for(;isspace(buff[i]) && buff[i]!='\0';i++);
	if(buff[i]=='\0'){
	    fprintf(stderr,"Empty value in parameter file (%d)\n",line);
	}

      /* set parameter and value */
      /*-------------------------*/
	set_param(buff,buff+i);
    }

    fclose(fd);
}


#define STRNCMP(s) (strncmp(param,s,strlen(s))==0 &&  \
		    (param[strlen(s)]=='\0' || isspace(param[strlen(s)])))


void
set_param(param,value)
char *param, *value;
{
    char l1[MAX_LABEL_LEN], l2[MAX_LABEL_LEN];

    if(STRNCMP("DEBUG")){

	DEBUG = atoi(value);

    }else if(STRNCMP("MAX_ERROR")){

	Max_error = atoi(value);

    }else if(STRNCMP("CUTOFF_LEN")){

	TOT_cut_len = atoi(value);

    }else if(STRNCMP("LABELED")){

	F_label = atoi(value);

    }else if(STRNCMP("DELETE_LABEL")){

	Delete_label[Delete_label_n] = (char *)malloc(strlen(value)+1);
	strcpy(Delete_label[Delete_label_n],value);
	Delete_label_n++;

    }else if(STRNCMP("DELETE_LABEL_FOR_LENGTH")){

	Delete_label_for_length[Delete_label_for_length_n] = (char *)malloc(strlen(value)+1);
	strcpy(Delete_label_for_length[Delete_label_for_length_n],value);
	Delete_label_for_length_n++;

    }else if(STRNCMP("QUOTE_LABEL")){

	Quote_term[Quote_term_n] = (char *)malloc(strlen(value)+1);
	strcpy(Quote_term[Quote_term_n],value);
	Quote_term_n++;

    }else if(STRNCMP("EQ_LABEL")){

	if(narg(value)!=2){
	    fprintf(stderr,"EQ_LABEL requires two values\n");
	    return;
	}
	sscanf(value,"%s %s",l1,l2);
	EQ_label[EQ_label_n].s1 = (char *)malloc(strlen(l1)+1);
	strcpy(EQ_label[EQ_label_n].s1,l1);
	EQ_label[EQ_label_n].s2 = (char *)malloc(strlen(l2)+1);
	strcpy(EQ_label[EQ_label_n].s2,l2);
	EQ_label_n++;

    }else if(STRNCMP("EQ_WORD")){

	if(narg(value)!=2){
	    fprintf(stderr,"EQ_WORD requires two values\n");
	    return;
	}
	sscanf(value,"%s %s",l1,l2);
	EQ_word[EQ_word_n].s1 = (char *)malloc(strlen(l1)+1);
	strcpy(EQ_word[EQ_word_n].s1,l1);
	EQ_word[EQ_word_n].s2 = (char *)malloc(strlen(l2)+1);
	strcpy(EQ_word[EQ_word_n].s2,l2);
	EQ_word_n++;

    }else{

	fprintf(stderr,"Unknown keyword (%s) in parameter file\n",param);

    }
}


int
narg(s)
char *s;
{
    int n;

    for(n=0;*s!='\0';){
	for(;isspace(*s);s++);
	if(*s=='\0'){
	    break;
	}
	n++;
	for(;!isspace(*s);s++){
	    if(*s=='\0'){
		break;
	    }
	}
    }

    return(n);
}

/*-----------------------------*/
/* Read line and gather data.  */
/* Return langth of sentence.  */
/*-----------------------------*/
int
read_line(buff, terminal, quotterm, wn, bracket, bn)
char *buff;
s_terminal terminal[];
s_term_ind quotterm[];
int *wn;
s_bracket bracket[];
int *bn;
{
    char *p, *q, label[MAX_LABEL_LEN], word[MAX_WORD_LEN];
    int	  qt;		  /* quote term counter */
    int   wid, bid;       /* word ID, bracket ID */
    int   n;              /* temporary remembering the position */
    int   b;              /* temporary remembering bid */
    int   i;
    int   len;            /* length of the sentence */

    len = 0;
    stack_top=0;

    for(p=buff,qt=0,wid=0,bid=0;*p!='\0';){

	if(isspace(*p)){
	    p++;
	    continue;

        /* open bracket */
        /*--------------*/
	}else if(*p=='('){

	    n=wid;
	    for(p++,i=0;!is_terminator(*p);p++,i++){
		label[i]=*p;
	    }
	    label[i]='\0';

	    /* Find terminals */
	    q = p;
	    if(isspace(*q)){
		for(q++;isspace(*q);q++);
		for(i=0;!is_terminator(*q);q++,i++){
		    word[i]=*q;
		}
		word[i]='\0';

                /* compute length */
		if(*q==')' && !is_deletelabel_for_length(label)==1){
		    len++;
		}
      if (DEBUG>1)
         printf("label=%s, word=%s, wid=%d\n",label,word,wid);
		/* quote terminal */
		if(*q==')' && is_quote_term(label,word)==1){
			strcpy(quotterm[qt].term.word,word);
			strcpy(quotterm[qt].term.label,label);
			quotterm[qt].index = wid;
         quotterm[qt].bracket = bid;
         quotterm[qt].endslen = stack_top;
         //quotterm[qt].ends = (int*)malloc(stack_top*sizeof(int));
         memcpy(quotterm[qt].ends,stack,stack_top*sizeof(int));
			qt++;
		}
		
                /* delete terminal */
		if(*q==')' && is_deletelabel(label)==1){
		    p = q+1;
		    continue;

		/* valid terminal */
		}else if(*q==')'){
		    strcpy(terminal[wid].word,word);
		    strcpy(terminal[wid].label,label);
		    wid++;
		    p = q+1;
		    continue;

                /* error */
		}else if(*q!='('){
		    Error("More than two elements in a bracket\n");
		}
	    }

            /* otherwise non-terminal label */
	    bracket[bid].start = wid;
	    bracket[bid].buf_start = p-buff;
	    strcpy(bracket[bid].label,label);
	    pushb(bid);
	    bid++;

	/* close bracket */
        /*---------------*/
	}else if(*p==')'){

	    b = popb();
	    bracket[b].end = wid;
	    bracket[b].buf_end = p-buff;
	    p++;

        /* error */
        /*-------*/
	}else{

	    Error("Reading sentence\n");
	}
    }

    if(!stackempty()){
	Error("Bracketing is unbalanced (too many open bracket)\n");
    }

    *wn = wid;
    *bn = bid;

    return(len);
}


/*----------------------*/
/* stack operation      */
/* for bracketing pairs */
/*----------------------*/
void
pushb(item)
int item;
{
    stack[stack_top++]=item;
}

int
popb()
{
    int item;

    item = stack[stack_top-1];

    if(stack_top-- < 0){
	Error("Bracketing unbalance (too many close bracket)\n");
    }
    return(item);
}

int
stackempty()
{
    if(stack_top==0){
	return(1);
    }else{
	return(0);
    }
}


/*------------------*/
/* calculate result */
/*------------------*/
void
calc_result(unsigned char *buf1,unsigned char *buf)
{
    int i, j, l;
    int match, crossing, correct_tag;

    int last_i = -1;

    char my_buf[1000];
    int match_found = 0;
    
    char match_j[200];
    for (j = 0; j < bn2; ++j) {
      match_j[j] = 0;
    }

    /* ML */
    if (DEBUG>1)
    	printf("\n");


    /* Find skip and error */
    /*---------------------*/
    if(wn2==0){
	Status = 2;
	individual_result(0,0,0,0,0,0);
	return;
    }

   if(wn1 != wn2){
      //if (DEBUG>1)
    //Error("Length unmatch (%d|%d)\n",wn1,wn2);
	fix_quote();
	if(wn1 != wn2){
		Error("Length unmatch (%d|%d)\n",wn1,wn2);
		individual_result(0,0,0,0,0,0);
		return;
	}
    }

    for(i=0;i<wn1;i++){
	if(word_comp(terminal1[i].word,terminal2[i].word)==0){
	    Error("Words unmatch (%s|%s)\n",terminal1[i].word,
                                            terminal2[i].word);
	    individual_result(0,0,0,0,0,0);
	    return;
	}
    }

    /* massage the data */
    /*------------------*/
    massage_data();
	   
    /* matching brackets */
    /*-------------------*/
    match = 0;
    for(i=0;i<bn1;i++){
      for(j=0;j<bn2;j++){

         if (DEBUG>1)
   printf("1.res=%d, 2.res=%d, 1.start=%d, 2.start=%d, 1.end=%d, 2.end=%d\n",bracket1[i].result,bracket2[j].result,bracket1[i].start,bracket2[j].start,bracket1[i].end,bracket2[j].end);

	// does bracket match?
	if(bracket1[i].result != 5 && 
	   bracket2[j].result == 0 &&
	   bracket1[i].start == bracket2[j].start && bracket1[i].end == bracket2[j].end) {

	  // (1) do we not care about the label or (2) does the label match?
	  if (F_label==0 || label_comp(bracket1[i].label,bracket2[j].label)==1) {
	    bracket1[i].result = bracket2[j].result = 1;
	    match++;
	    match_found = 1;
	    break;
	  } else {
	    if (DEBUG>1) {
	      printf("  LABEL[%d-%d]: ",bracket1[i].start,bracket1[i].end-1);
	      l = bracket1[i].buf_end-bracket1[i].buf_start;
	      strncpy(my_buf,buf1+bracket1[i].buf_start,l);
	      my_buf[l] = '\0';
	      printf("%s\n",my_buf);
	    }
	    match_found = 1;
	    match_j[j] = 1;
	  }
	}
      }

      if (!match_found && bracket1[i].result != 5 && DEBUG>1) {
	/* ### ML 09/28/03: gold bracket with no corresponding test bracket */
	printf("  BRACKET[%d-%d]: ",bracket1[i].start,bracket1[i].end-1);
	l = bracket1[i].buf_end-bracket1[i].buf_start;
	strncpy(my_buf,buf1+bracket1[i].buf_start,l);
	my_buf[l] = '\0';
	printf("%s\n",my_buf);
      }
      match_found = 0;
    }

    for(j=0;j<bn2;j++){
      if (bracket2[j].result==0 && !match_j[j] && DEBUG>1) {
	/* test bracket with no corresponding gold bracket */
	printf("  EXTRA[%d-%d]: ",bracket2[j].start,bracket2[j].end-1);
	l = bracket2[j].buf_end-bracket2[j].buf_start;
	strncpy(my_buf,buf+bracket2[j].buf_start,l);
	my_buf[l] = '\0';
	printf("%s\n",my_buf);
      }
    }

    /* crossing */
    /*----------*/
    crossing = 0;

    /* crossing is counted based on the brackets */
    /* in test rather than gold file (by Mike)   */
    for(j=0;j<bn2;j++){
      for(i=0;i<bn1;i++){
	if(bracket1[i].result != 5 &&
	   bracket2[j].result != 5 &&
	   ((bracket1[i].start < bracket2[j].start &&
	     bracket1[i].end   > bracket2[j].start &&
	     bracket1[i].end   < bracket2[j].end) ||
	    (bracket1[i].start > bracket2[j].start &&
	     bracket1[i].start < bracket2[j].end &&
	     bracket1[i].end   > bracket2[j].end))){

	  /* ### ML 09/01/03: get details on cross-brackettings */
	  if (i != last_i) {
	    if (DEBUG>1) {
	    	printf("  CROSSING[%d-%d]: ",bracket1[i].start,bracket1[i].end-1);
	    	l = bracket1[i].buf_end-bracket1[i].buf_start;
	    	strncpy(my_buf,buf1+bracket1[i].buf_start,l);
	    	my_buf[l] = '\0';
	    	printf("%s\n",my_buf);

	    	/* ML
	    	printf("\n  CROSSING at bracket %d:\n",i-1);
	    	printf("  GOLD (tokens %d-%d): ",bracket1[i].start,bracket1[i].end-1);
	    	l = bracket1[i].buf_end-bracket1[i].buf_start;
	    	strncpy(my_buf,buf1+bracket1[i].buf_start,l);
	    	my_buf[l] = '\0';
	    	printf("%s\n",my_buf);
	    	*/
	    }
	    last_i = i;
	  }

	  /* ML
	  printf("  TEST (tokens %d-%d): ",bracket2[j].start,bracket2[j].end-1);
	  l = bracket2[j].buf_end-bracket2[j].buf_start;
	  strncpy(my_buf,buf+bracket2[j].buf_start,l);
	  my_buf[l] = '\0';
	  printf("%s\n",my_buf);
	  */

	  crossing++;
	  break;
	}
      }
    }

    /* Tagging accuracy */
    /*------------------*/
    correct_tag=0;
    for(i=0;i<wn1;i++){
       if(label_comp(terminal1[i].label,terminal2[i].label)==1){
          terminal1[i].result = terminal2[i].result = 1;
          correct_tag++;
       } else {
          terminal1[i].result = terminal2[i].result = 0;
       }
    }

    individual_result(wn1,r_bn1,r_bn2,match,crossing,correct_tag);
}

void
fix_quote()
{
   int i,j,k;
   if (DEBUG>1) {
      for(i=0;i<MAX_QUOTE_TERM;i++){
         if (quotterm1[i].index!=-1)
            printf("%d: %s - %s\n",quotterm1[i].index,
                  quotterm1[i].term.label,
                  quotterm1[i].term.word);
         if (quotterm2[i].index!=-1)
            printf("%d: %s - %s\n",quotterm2[i].index,
                  quotterm2[i].term.label,
                  quotterm2[i].term.word);
      }
   }
   for(i=0;i<MAX_QUOTE_TERM;i++) {
      int ind = quotterm2[i].index;
      if (ind!=-1) {
         for(j=0;j<MAX_QUOTE_TERM;j++){
            if (quotterm1[j].index==ind &&
                  strcmp(quotterm1[j].term.label,
                     quotterm2[i].term.label)!=0) {
               if (is_deletelabel(quotterm1[j].term.label) && !is_deletelabel(quotterm2[i].term.label)) {
                  reinsert_term(&quotterm1[j],terminal1,bracket1,&wn1);
                  for (k=j;k<MAX_QUOTE_TERM;k++)
                     if (quotterm1[k].index!=-1)
                        quotterm1[k].index++;
               } else if (is_deletelabel(quotterm2[i].term.label) && !is_deletelabel(quotterm1[j].term.label)) {
                  reinsert_term(&quotterm2[i],terminal2,bracket2,&wn2);
                  for (k=i;k<MAX_QUOTE_TERM;k++)
                     if (quotterm2[k].index!=-1)
                        quotterm2[k].index++;
               }
            }
         }
      } else break;
   }
}

void
reinsert_term(quot,terminal,bracket,wn)
s_term_ind* quot;
s_terminal terminal[];
s_bracket bracket[];
int* wn;
{
   int ind = quot->index;
   int bra = quot->bracket;
   s_terminal* term = &quot->term;
   int k;
   memmove(&terminal[ind+1],
         &terminal[ind],
         sizeof(s_terminal)*(MAX_WORD_IN_SENT-ind-1));
   strcpy(terminal[ind].label,term->label);
   strcpy(terminal[ind].word,term->word);
   (*wn)++;
   if (DEBUG>1)
      printf("bra=%d, ind=%d\n",bra,ind);
   for(k=0;k<MAX_BRACKET_IN_SENT;k++) {
      if (bracket[k].start==-1)
         break;
      if (DEBUG>1)
         printf("bracket[%d]={%d,%d}\n",k,bracket[k].start,bracket[k].end);
      if (k>=bra) {
         bracket[k].start++;
         bracket[k].end++;
      }
      //if (bracket[k].start<=ind && bracket[k].end>=ind)
         //bracket[k].end++;
   }
   if (DEBUG>1)
      printf("endslen=%d\n",quot->endslen);
   for(k=0;k<quot->endslen;k++) {
      //printf("ends[%d]=%d",k,quot->ends[k]);
      bracket[quot->ends[k]].end++;
   }
   //free(quot->ends);
}
/*
void
adjust_end(ind,bra)
int ind;
int bra;
{
    for(k=0;k<MAX_BRACKET_IN_SENT;k++) {
      if (bracket[k].start==-1)
         break;
      printf("bracket[%d]={%d,%d}\n",k,bracket[k].start,bracket[k].end);
      if (k>=bra)
         bracket[k].end++;
   }
}
*/
void
massage_data()
{
    int i, j;

    /* for GOLD */
    /*----------*/ 
    for(i=0;i<bn1;i++){

	bracket1[i].result = 0;

	/* Zero element */
	if(bracket1[i].start == bracket1[i].end){
	    bracket1[i].result = 5;
	    continue;
	}

        /* Modify label */
	modify_label(bracket1[i].label);

	/* Delete label */
	for(j=0;j<Delete_label_n;j++){
	    if(label_comp(bracket1[i].label,Delete_label[j])==1){
		bracket1[i].result = 5;
	    }
	}
    }
	   
    /* for TEST */
    /*----------*/ 
    for(i=0;i<bn2;i++){

	bracket2[i].result = 0;

	/* Zero element */
	if(bracket2[i].start == bracket2[i].end){
	    bracket2[i].result = 5;
	    continue;
	}

        /* Modify label */
	modify_label(bracket2[i].label);

	/* Delete label */
	for(j=0;j<Delete_label_n;j++){
	    if(label_comp(bracket2[i].label,Delete_label[j])==1){
		bracket2[i].result = 5;
	    }
	}
    }


    /* count up real number of brackets (exclude deleted ones) */
    /*---------------------------------------------------------*/
    r_bn1 = r_bn2 = 0;

    for(i=0;i<bn1;i++){
	if(bracket1[i].result != 5){
	    r_bn1++;
	}
    }

    for(i=0;i<bn2;i++){
	if(bracket2[i].result != 5){
	    r_bn2++;
	}
    }
}


/*------------------------*/
/* trim the tail of label */
/*------------------------*/
void
modify_label(label)
char *label;
{
    char *p;

    for(p=label;*p!='\0';p++){
	if(*p=='-' || *p=='='){
	    *p='\0';
	    break;
	}
    }
}


/*-----------------------------------------------*/
/* add individual statistics to TOTAL statictics */
/*-----------------------------------------------*/
void
individual_result(wn1,bn1,bn2,match,crossing,correct_tag)
int wn1,bn1,bn2,match,crossing,correct_tag;
{

    /* Statistics for ALL */
    /*--------------------*/
    TOTAL_sent++;
    if(Status==1){
	TOTAL_error_sent++;
    }else if(Status==2){
	TOTAL_skip_sent++;
    }else{
	TOTAL_bn1 += bn1;
	TOTAL_bn2 += bn2;
	TOTAL_match += match;
	if(bn1==bn2 && bn2==match){
	    TOTAL_comp_sent++;
	}
	TOTAL_word += wn1;
	TOTAL_crossing += crossing;
	if(crossing==0){
	    TOTAL_no_crossing++;
	}
	if(crossing <= 2){
	    TOTAL_2L_crossing++;
	}
	TOTAL_correct_tag += correct_tag;
    }


    /* Statistics for sent length <= TOT_cut_len */
    /*-------------------------------------------*/
    if(r_wn1<=TOT_cut_len){
	TOT40_sent++;
	if(Status==1){
	    TOT40_error_sent++;
	}else if(Status==2){
	    TOT40_skip_sent++;
	}else{
	    TOT40_bn1 += bn1;
	    TOT40_bn2 += bn2;
	    TOT40_match += match;
	    if(bn1==bn2 && bn2==match){
		TOT40_comp_sent++;
	    }
	    TOT40_word += wn1;
	    TOT40_crossing += crossing;
	    if(crossing==0){
		TOT40_no_crossing++;
	    }
	    if(crossing <= 2){
		TOT40_2L_crossing++;
	    }
	    TOT40_correct_tag += correct_tag;
	}
    }

    /* Print individual result */
    /*-------------------------*/
    printf("%4d  %3d    %d  ",Line,r_wn1,Status);
    printf("%6.2f %6.2f   %3d    %3d  %3d    %3d",
	   (r_bn1==0?0.0:100.0*match/r_bn1), 
	   (r_bn2==0?0.0:100.0*match/r_bn2),
	   match, r_bn1, r_bn2, crossing);

    printf("   %4d  %4d   %6.2f\n",wn1,correct_tag,
	   (wn1==0?0.0:100.0*correct_tag/wn1));
}


/*------------------------*/
/* print total statistics */
/*------------------------*/
void
print_total()
{
    int sentn;
    double r,p,f;

    printf("============================================================================\n");

    if(TOTAL_bn1>0 && TOTAL_bn2>0){
	printf("                %6.2f %6.2f %6d %5d %5d  %5d",
	       (TOTAL_bn1>0?100.0*TOTAL_match/TOTAL_bn1:0.0),
	       (TOTAL_bn2>0?100.0*TOTAL_match/TOTAL_bn2:0.0),
	       TOTAL_match, 
	       TOTAL_bn1, 
	       TOTAL_bn2,
	       TOTAL_crossing);
    }

    printf("  %5d %5d   %6.2f",
	   TOTAL_word,
	   TOTAL_correct_tag,
	   (TOTAL_word>0?100.0*TOTAL_correct_tag/TOTAL_word:0.0));

    printf("\n");
    printf("=== Summary ===\n");

    sentn = TOTAL_sent - TOTAL_error_sent - TOTAL_skip_sent;

    printf("\n-- All --\n");
    printf("Number of sentence        = %6d\n",TOTAL_sent);
    printf("Number of Error sentence  = %6d\n",TOTAL_error_sent);
    printf("Number of Skip  sentence  = %6d\n",TOTAL_skip_sent);
    printf("Number of Valid sentence  = %6d\n",sentn);
    
    r = TOTAL_bn1>0 ? 100.0*TOTAL_match/TOTAL_bn1 : 0.0;
    printf("Bracketing Recall         = %6.2f\n",r);

    p = TOTAL_bn2>0 ? 100.0*TOTAL_match/TOTAL_bn2 : 0.0;
    printf("Bracketing Precision      = %6.2f\n",p);

    f = 2*p*r/(p+r);
    printf("Bracketing FMeasure       = %6.2f\n",f);
			    
    printf("Complete match            = %6.2f\n",
	   (sentn>0?100.0*TOTAL_comp_sent/sentn:0.0));
    printf("Average crossing          = %6.2f\n",
	   (sentn>0?1.0*TOTAL_crossing/sentn:0.0));
    printf("No crossing               = %6.2f\n",
	   (sentn>0?100.0*TOTAL_no_crossing/sentn:0.0));
    printf("2 or less crossing        = %6.2f\n",
	   (sentn>0?100.0*TOTAL_2L_crossing/sentn:0.0));
    printf("Tagging accuracy          = %6.2f\n",
	   (TOTAL_word>0?100.0*TOTAL_correct_tag/TOTAL_word:0.0));

    sentn = TOT40_sent - TOT40_error_sent - TOT40_skip_sent;

    printf("\n-- len<=%d --\n",TOT_cut_len);
    printf("Number of sentence        = %6d\n",TOT40_sent);
    printf("Number of Error sentence  = %6d\n",TOT40_error_sent);
    printf("Number of Skip  sentence  = %6d\n",TOT40_skip_sent);
    printf("Number of Valid sentence  = %6d\n",sentn);


    r = TOT40_bn1>0 ? 100.0*TOT40_match/TOT40_bn1 : 0.0;
    printf("Bracketing Recall         = %6.2f\n",r);

    p = TOT40_bn2>0 ? 100.0*TOT40_match/TOT40_bn2 : 0.0;
    printf("Bracketing Precision      = %6.2f\n",p);

    f = 2*p*r/(p+r);
    printf("Bracketing FMeasure       = %6.2f\n",f);

    printf("Complete match            = %6.2f\n",
	   (sentn>0?100.0*TOT40_comp_sent/sentn:0.0));
    printf("Average crossing          = %6.2f\n",
	   (sentn>0?1.0*TOT40_crossing/sentn:0.0));
    printf("No crossing               = %6.2f\n",
	   (sentn>0?100.0*TOT40_no_crossing/sentn:0.0));
    printf("2 or less crossing        = %6.2f\n",
	   (sentn>0?100.0*TOT40_2L_crossing/sentn:0.0));
    printf("Tagging accuracy          = %6.2f\n",
	   (TOT40_word>0?100.0*TOT40_correct_tag/TOT40_word:0.0));

}


/*--------------------------------*/
/* display individual information */
/*--------------------------------*/
void
dsp_info()
{
  int i, n;

  printf("-<1>---(wn1=%3d, bn1=%3d)-           ",wn1,bn1);
  printf("-<2>---(wn2=%3d, bn2=%3d)-\n",wn2,bn2);

  n = (wn1>wn2?wn1:wn2);

  for(i=0;i<n;i++){
      if(terminal1[i].word[0]!='\0'){
	  printf("%3d : %d : %-6s  %-16s      ",i,terminal1[i].result,
		 terminal1[i].label,terminal1[i].word);
      }else{
	  printf("                                        ");
      }

      if(terminal2[i].word[0]!='\0'){
	  printf("%3d : %d : %-6s  %-16s\n",i,terminal2[i].result,
		 terminal2[i].label,terminal2[i].word);
      }else{
	  printf("\n");
      }
  }
  printf("\n");

  n = (bn1>bn2?bn1:bn2);

  for(i=0;i<n;i++){
      if(bracket1[i].start != -1){
	  printf("%3d : %d : %3d  %3d  %-6s      ",i,bracket1[i].result,
	                            bracket1[i].start,bracket1[i].end,
	                            bracket1[i].label);
      } else {
	  printf("                                ");
      }

      if(bracket2[i].start != -1){
	  printf("%3d : %d : %3d  %3d  %-6s\n",i,bracket2[i].result,
                                    bracket2[i].start,bracket2[i].end,
	                            bracket2[i].label);
      } else {
	  printf("\n");
      }
  }
  printf("\n");

  printf("========\n");

}


/*-----------------*/
/* some predicates */
/*-----------------*/
int
is_terminator(c)
char c;
{
    if(isspace(c) || c=='(' || c==')'){
	return(1);
    }else{
	return(0);
    }
}

int
is_deletelabel(s)
char *s;
{
    int i;

    for(i=0;i<Delete_label_n;i++){
	if(strcmp(s,Delete_label[i])==0){
	    return(1);
	}
    }

    return(0);
}

int
is_deletelabel_for_length(s)
char *s;
{
    int i;

    for(i=0;i<Delete_label_for_length_n;i++){
	if(strcmp(s,Delete_label_for_length[i])==0){
	    return(1);
	}
    }

    return(0);
}

int
is_quote_term(s,w)
char *s;
char *w;
{
    int i;

    for(i=0;i<Quote_term_n;i++){
		if(strcmp(s,Quote_term[i])==0){
			if (strcmp(w,"'")==0 || strcmp(w,"\"")==0 || strcmp(w,"/")==0)
	    	return(1);
		}
    }

    return(0);
}


/*---------------*/
/* compare words */
/*---------------*/
int
word_comp(s1,s2)
char *s1,*s2;
{
    int i;

    if(strcmp(s1,s2)==0){
	return(1);
    }

    for(i=0;i<EQ_word_n;i++){
	if((strcmp(s1,EQ_word[i].s1)==0 &&
	    strcmp(s2,EQ_word[i].s2)==0) ||
	   (strcmp(s1,EQ_word[i].s2)==0 &&
	    strcmp(s2,EQ_word[i].s1)==0)){
	    return(1);
	}
    }

    return(0);
}

/*----------------*/
/* compare labels */
/*----------------*/
int
label_comp(s1,s2)
char *s1,*s2;
{
    int i;

    if(strcmp(s1,s2)==0){
	return(1);
    }

    for(i=0;i<EQ_label_n;i++){
	if((strcmp(s1,EQ_label[i].s1)==0 &&
	    strcmp(s2,EQ_label[i].s2)==0) ||
	   (strcmp(s1,EQ_label[i].s2)==0 &&
	    strcmp(s2,EQ_label[i].s1)==0)){
	    return(1);
	}
    }

    return(0);
}


/*--------*/
/* errors */
/*--------*/
void
Error(s,arg1,arg2,arg3)
char *s, *arg1, *arg2, *arg3;
{
    Status = 1;
    fprintf(stderr,"%d : ",Line);
    fprintf(stderr,s,arg1,arg2,arg3);
    if(Error_count++>Max_error){
	exit(1);
    }
}


/*---------------------*/
/* fatal error to exit */
/*---------------------*/
void
Fatal(s,arg1,arg2,arg3)
char *s, *arg1, *arg2, *arg3;
{
    fprintf(stderr,s,arg1,arg2,arg3);
    exit(1);
}


/*-------*/
/* Usage */
/*-------*/
void
Usage()
{
  fprintf(stderr," evalb [-dDh][-c n][-e n][-p param_file] gold-file test-file  \n");
  fprintf(stderr,"                                                         \n");
  fprintf(stderr,"    Evaluate bracketing in test-file against gold-file.  \n");
  fprintf(stderr,"    Return recall, precision, F-Measure, tag accuracy.              \n");
  fprintf(stderr,"                                                         \n");
  fprintf(stderr,"  <option>                                               \n");
  fprintf(stderr,"    -d             debug mode                            \n");
  fprintf(stderr,"    -D             debug mode plus bracketing info       \n");
  fprintf(stderr,"    -c n           cut-off length forstatistics (def.=40)\n");
  fprintf(stderr,"    -e n           number of error to kill (default=10)  \n");
  fprintf(stderr,"    -p param_file  parameter file                        \n");
  fprintf(stderr,"    -h    help                                           \n");
}
