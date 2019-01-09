//Yeditech
//pcostasgr 2018 Simple Neural Net 
//Based on https://causeyourestuck.io/2017/06/12/neural-network-scratch-theory
//Data are from MNIST database http://yann.lecun.com/exdb/mnist/
//
extern crate rand;
extern crate byteorder;

use self::rand::Rng;
use std::io;
use std::io::prelude::*;
use std::fs::{File,OpenOptions};
use std::fmt::Debug;
use std::time::{Duration, Instant};
use std::f32;
use std::io::Cursor;
use byteorder::{ByteOrder, LittleEndian};

#[derive(Debug)]
pub struct Mat<T> {
    width:usize,
    height:usize,
    data:Vec<T>, 
}


impl<T:Debug > Mat<T > {
    
    pub fn debug_matrix(&self,msg:&str){

        let width:usize=self.width;
        let height:usize=self.height;
        let mut index:usize=0;

        println!("Debugging");    
        println!("{}",msg);
        println!("--------------------------------------------------------");
        println!("height { }",height);
        println!("width  { }",width);

        for y in 0..height {
            println!("");
            for x in 0..width {

                index=width*y+x;

                print!("{:?} ",self.data[index]);
            }
        }
        println!("");
        println!("");
    }
    
    pub fn matrix_info(&self,name:&str){

        let width:usize=self.width;
        let height:usize=self.height;
        let mut index:usize=0;

        println!("info {}",name);    
        println!("--------------------------------------------------------");
        println!("height { }",height);
        println!("width  { }",width);

        println!("");
    }

}
//-------------------------------------------------------------------------------------------------------------------------
type Matrix=Mat<f32>;
type ImData=Mat<u8>;


impl Matrix{
    pub fn new(rows:usize,columns:usize)->Matrix{
        Matrix{
            width:columns,
            height:rows,
            data:vec![0.0;rows*columns]
        }
    }

    pub fn new2(rows:usize,columns:usize,input:&Vec<f32>)->Matrix {
        Matrix{
            width:columns,
            height:rows,
            data:input.clone()
        }
    }

    pub fn mul_matrix_scalar(& mut self,scalar:f32 ) {
                                            
        let mut index:usize=0;
        
        for y in 0..self.height {
            for x in 0..self.width {
                index=self.width*y+x;
                self.data[index]*=scalar;
            }
        }
   
    }

}
//--------------------------------------------------------------------------------------------------------------------------
pub struct NeuralNet{
    input_layers:usize,
    hidden_layers:usize,
    output_layers:usize,
    learning_rate:f32,
    W1:Matrix,
    W2:Matrix,
    B1:Matrix,
    B2:Matrix,
    H:Matrix,
    Y:Matrix,
    X:Matrix
}
//--------------------------------------------------------------------------------------------------------------------------
impl NeuralNet {

    pub fn new(input_layer_no:usize,hidden_layer_no:usize,output_layer_no:usize,learning_rate:f32) -> NeuralNet{
        
       NeuralNet {
            input_layers:input_layer_no,
            hidden_layers:hidden_layer_no,
            output_layers:output_layer_no,               
            learning_rate:learning_rate,
            W1:Matrix::new(input_layer_no,hidden_layer_no),
            W2:Matrix::new(hidden_layer_no,output_layer_no),
            B1:Matrix::new(1,hidden_layer_no),
            B2:Matrix::new(1,output_layer_no),
            X:Matrix::new(1,input_layer_no),
            H:Matrix::new(1,hidden_layer_no),
            Y:Matrix::new(1,output_layer_no)
       }

    }

    pub fn init_net(&mut self){
        
        let mut rng=rand::thread_rng();

        self.W1.data=self.W1.data.iter_mut().map( |x| rng.gen_range(0.0,1.0) ).collect();
        self.W2.data=self.W2.data.iter_mut().map( |x| rng.gen_range(0.0,1.0) ).collect();
        self.B1.data=self.B1.data.iter_mut().map( |x| rng.gen_range(0.0,1.0) ).collect();
        self.B2.data=self.B2.data.iter_mut().map( |x| rng.gen_range(0.0,1.0) ).collect();
        
    }
//--------------------------------------------------------------------------------------------------------------------------
  pub fn read_from_file(& mut self,filename:&str){
       let mut f=match OpenOptions::new()
            .read(true)
            .open(filename){
                    Ok(file)=> file,
                    Err(..) => panic!("Error reading file:{}",filename)
        
            };
    
    let mut bu64:[u8;8]=[0;8];
    
    f.read(& mut bu64);
    self.input_layers=LittleEndian::read_u64(& mut bu64) as usize;

    f.read(& mut bu64);
    self.hidden_layers=LittleEndian::read_u64(& mut bu64) as usize;

    f.read(& mut bu64);
    self.output_layers=LittleEndian::read_u64(& mut bu64) as usize;


    let mut learning_rate:u64=0;

    f.read(& mut bu64);
    learning_rate=LittleEndian::read_u64(& mut bu64);

    self.learning_rate=learning_rate as f32 /100.0 as f32;

    println!("W1");

    let mut height:usize=0;
    let mut width:usize=0;
    let mut total_size:usize=0;
    
    //W1
    //---------------------------------------------------------------------------------------------------------------------------
    f.read(& mut bu64);
    height=LittleEndian::read_u64(& mut bu64) as usize;

    f.read(& mut bu64);
    width=LittleEndian::read_u64(& mut bu64) as usize;
   
    println!("W1 height:{} width:{}",height,width);
    total_size=width*height;
    let mut w1_buffer:Vec<u8>=vec![0;total_size*4];


    self.W1=Matrix{
                width:width,
                height:height,
                data:vec![0.0;total_size]                    
            };
   
     f.read(& mut w1_buffer);
     LittleEndian::read_f32_into_unchecked(&w1_buffer, &mut self.W1.data);
    

    println!("W2");

    //W2
    //---------------------------------------------------------------------------------------------------------------------------
    f.read(& mut bu64);
    height=LittleEndian::read_u64(& mut bu64) as usize;

    f.read(& mut bu64);
    width=LittleEndian::read_u64(& mut bu64) as usize;
    
    println!("W2 height:{} width:{}",height,width);

    total_size=width*height;
    let mut w2_buffer:Vec<u8>=vec![0;total_size*4];


    self.W2=Matrix{
                width:width,
                height:height,
                data:vec![0.0;total_size]                    
            };
    
    f.read(& mut w2_buffer);
    LittleEndian::read_f32_into_unchecked(&w2_buffer, &mut self.W2.data);

    println!("B1");
    //B1
    //---------------------------------------------------------------------------------------------------------------------------
    f.read(& mut bu64);
    height=LittleEndian::read_u64(& mut bu64) as usize;

    f.read(& mut bu64);
    width=LittleEndian::read_u64(& mut bu64) as usize;
    
    println!("B1 height:{} width:{}",height,width);

    total_size=width*height;
    let mut b1_buffer:Vec<u8>=vec![0;total_size*4];

    self.B1=Matrix{
                width:width,
                height:height,
                data:vec![0.0;total_size]                    
            };
    
    f.read(& mut b1_buffer);
    LittleEndian::read_f32_into_unchecked(&b1_buffer, &mut self.B1.data);

    println!("B2");

    //B2
    //---------------------------------------------------------------------------------------------------------------------------
    f.read(& mut bu64);
    height=LittleEndian::read_u64(& mut bu64) as usize;

    f.read(& mut bu64);
    width=LittleEndian::read_u64(& mut bu64) as usize;
   
    println!("B2 height:{} width:{}",height,width);
    total_size=width*height;
    let mut b2_buffer:Vec<u8>=vec![0;total_size*4];


    self.B2=Matrix{
                width:width,
                height:height,
                data:vec![0.0;total_size]                    
            };
    
    f.read(& mut b2_buffer);
    LittleEndian::read_f32_into_unchecked(&b2_buffer, &mut self.B2.data);


    println!("Read data");
    println!("-------------------------------------------------------");
    println!("input:{}",self.input_layers);
    println!("hidden:{}",self.hidden_layers );
    println!("output:{}",self.output_layers);
    println!("rate  :{}",self.learning_rate);


    //self.W1.debug_matrix("W1");
    //self.W2.debug_matrix("W2");
 //   self.B2.debug_matrix();
  }
//--------------------------------------------------------------------------------------------------------------------------
    pub fn save_to_file(&self,filename:&str){

        let mut f=match OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(filename){
                    Ok(file)=> file,
                    Err(..) => panic!("Error creating/opening file:{}",filename)
        
            };
       
    //let vu=to_byte_slice_f32(&vf);

    //println!("slice : {:?}",vu);
    
        println!("Save data");
        println!("-------------------------------------------------------");
        println!("input:{}",self.input_layers);
        println!("hidden:{}",self.hidden_layers );
        println!("output:{}",self.output_layers);
        println!("rate  :{}",self.learning_rate);

        let mut bu64:[u8;8]=[0;8];


        LittleEndian::write_u64(& mut bu64,self.input_layers as u64);
        f.write(&bu64);

        LittleEndian::write_u64(& mut bu64,self.hidden_layers as u64);
        f.write(&bu64);

        LittleEndian::write_u64(& mut bu64,self.output_layers as u64);
        f.write(&bu64);
        
        let learning_rate=(self.learning_rate*100.0) as usize;

        LittleEndian::write_u64(& mut bu64,learning_rate as u64);
        f.write(&bu64);
        
        //W1
        let mut size_=self.W1.height*self.W1.width;
        LittleEndian::write_u64(& mut bu64,self.W1.height as u64);
        f.write(&bu64);

        LittleEndian::write_u64(& mut bu64,self.W1.width as u64);
        f.write(&bu64);

        let w1_buffer_data=to_byte_slice_f32(&self.W1.data,4);
        f.write(&w1_buffer_data);

        //W2
        size_=self.W2.height*self.W2.width;
        LittleEndian::write_u64(& mut bu64,self.W2.height as u64);
        f.write(&bu64);

        LittleEndian::write_u64(& mut bu64,self.W2.width as u64);
        f.write(&bu64);
        
        
        let w2_buffer_data=to_byte_slice_f32(&self.W2.data,4);
        f.write(&w2_buffer_data);

        //B1
        size_=self.B1.height*self.B1.width;
        LittleEndian::write_u64(& mut bu64,self.B1.height as u64);
        f.write(&bu64);

        LittleEndian::write_u64(& mut bu64,self.B1.width as u64);
        f.write(&bu64);

        let b1_buffer_data=to_byte_slice_f32(&self.B1.data,4);
        f.write(&b1_buffer_data);

        //B2
        size_=self.B2.height*self.B2.width;
        LittleEndian::write_u64(& mut bu64,self.B2.height as u64);
        f.write(&bu64);

        LittleEndian::write_u64(& mut bu64,self.B2.width as u64);
        f.write(&bu64);

        let b2_buffer_data=to_byte_slice_f32(&self.B2.data,4);
        f.write(&b2_buffer_data);


        //self.W1.debug_matrix("W1");
        //self.W2.debug_matrix("W2");
        //self.B2.debug_matrix();
    }
//------------------------------------------------------------------------------------------------------------------------
    pub fn compute_output(&mut self,v:&Vec<f32>) {
        self.X.data=v.to_vec();

        let mut m1=mul_matrices(&self.X,&self.W1);

        self.H=add_matrices(&m1,&self.B1);

        self.H.data.iter_mut().for_each( |x| *x= sigmoid(*x));

        let mut m2=mul_matrices(&self.H,&self.W2);

        self.Y=add_matrices(&m2,&self.B2);

        self.Y.data.iter_mut().for_each( |x| *x= sigmoid(*x));

    }
//------------------------------------------------------------------------------------------------------------------------
    pub fn learn(&mut self,input:&Vec<f32>) {
        
        let mut dJdW1=Matrix::new(self.input_layers,self.hidden_layers);
        let mut dJdW2=Matrix::new(self.hidden_layers,self.output_layers);
        let mut dJdB1=Matrix::new(1,self.hidden_layers);
        let mut dJdB2=Matrix::new(1,self.output_layers);
        
        let YO=Matrix::new2(1,self.output_layers,input);

        let mut YT=sub_matrices(&self.Y,&YO);

        let mut HW2=mul_matrices(&self.H,&self.W2);
        
        let mut HW2B2=add_matrices(&HW2,&self.B2);
        
        HW2B2.data.iter_mut().for_each( |x| *x=sigmoid_deriv(*x) );

        mul_matrices_val(&YT,&HW2B2,& mut dJdB2);

//----------------------------------------------------------------------------------------------------------------------        
        //println!("dJdB2");
        //dJdB2.debug_matrix();

        //println!("H");
        //self.H.debug_matrix();
//----------------------------------------------------------------------------------------------------------------------        
        
        let mut W2T=transpose_matrix(&self.W2);

        let mut dJdB2W2T=mul_matrices(&dJdB2,&W2T);

        let mut XW1=mul_matrices(&self.X,&self.W1);

        let mut XW1B1=add_matrices(&XW1,&self.B1);
        XW1B1.data.iter_mut().for_each( |x| *x=sigmoid_deriv(*x) );

        mul_matrices_val(&dJdB2W2T,&XW1B1,& mut dJdB1);

        //first occerence of NAN due to hgh values in dJdB1
;
        let mut HT=transpose_matrix(&self.H);
        let mut XT=transpose_matrix(&self.X);
        
        //dJdB1 is NAN from second iteration
        //
        dJdW2=mul_matrices(&HT,&dJdB2);
        dJdW1=mul_matrices(&XT,&dJdB1);


        let mut NW1=sub_matrices(&self.W1,&dJdW1);
        NW1.mul_matrix_scalar(self.learning_rate);

        let mut NW2=sub_matrices(&self.W2,&dJdW2);
        NW2.mul_matrix_scalar(self.learning_rate);

        let mut NB1=sub_matrices(&self.B1,&dJdB1);
        NB1.mul_matrix_scalar(self.learning_rate);

        let mut NB2=sub_matrices(&self.B2,&dJdB2);
        NB2.mul_matrix_scalar(self.learning_rate);

        self.W1.data.clear();
        self.W1.data=NW1.data.clone();

        self.W2.data.clear();
        self.W2.data=NW2.data.clone();

        self.B1.data.clear();
        self.B1.data=NB1.data.clone();

        self.B2.data.clear();
        self.B2.data=NB2.data.clone();




    }
}
//--------------------------------------------------------------------------------------------------------------------------
pub fn debug_matrix(matrix:& Matrix){

    let width:usize=matrix.width;
    let height:usize=matrix.height;
    let mut index:usize=0;
    let mut value:f32=0.0;

    println!("Debugging");    
    println!("--------------------------------------------------------");
    println!("height { }",height);
    println!("width  { }",width);

    for y in 0..height {
        println!("");
        for x in 0..width {

            index=width*y+x;
            value=matrix.data[index];

            print!("{ } ",value);
        }
    }
    println!("");
    println!("");
}
//--------------------------------------------------------------------------------------------------------------------------
pub fn add_matrices(m1:& Matrix,m2:& Matrix)->Matrix {
                                            
    let width:usize=m1.width;
    let height:usize=m1.height;
    let mut index:usize=0;
    let mut data:Vec<f32>=vec![0.0;width*height];

    if m1.width != m2.width || m1.height!= m2.height {
        panic!("add_matrices -> Matrices are not of the same dimensions");
    }


    for y in 0..height {
        for x in 0..width {
            index=width*y+x;
            data[index]=m1.data[index]+m2.data[index];
        }
    }
   
    Matrix{
        width:width,
        height:height,
        data:data.clone()
    }
}

//--------------------------------------------------------------------------------------------------------------------------
pub fn sub_matrices(m1:& Matrix,m2:& Matrix)->Matrix {
                                            
    let width:usize=m1.width;
    let height:usize=m1.height;
    let mut index:usize=0;
    let mut data:Vec<f32>=vec![0.0;width*height];

    if m1.width != m2.width || m1.height!= m2.height {
        panic!("sub_matrices -> Matrices are not of the same dimensions");
    }


    for y in 0..height {
        for x in 0..width {
            index=width*y+x;
            data[index]=m1.data[index]-m2.data[index];
        }
    }
   
    Matrix{
        width:width,
        height:height,
        data:data.clone()
    }
}
//--------------------------------------------------------------------------------------------------------------------------
pub fn mul_matrices_val(m1:& Matrix,m2:& Matrix, m:&mut Matrix ) {
                                            
    let width:usize=m1.width;
    let height:usize=m1.height;
    let mut index:usize=0;

    if m1.width != m2.width || m2.height!= m1.height {
        panic!("mul_matrices_val -> Matrices are not of the same dimensions");
    }

    if m1.width != m.width || m1.height!= m.height {
        panic!("mul_matrices_val -> Result Matrix does not have same dimensions as the inputs");
    }

    for y in 0..height {
        for x in 0..width {
            index=width*y+x;
            m.data[index]=m1.data[index]*m2.data[index];
        }
    }
   
}
//--------------------------------------------------------------------------------------------------------------------------
pub fn transpose_matrix(m1:& Matrix)->Matrix{

    let mut index1:usize=0;
    let mut index2:usize=0;

    let mut data:Vec<f32>=vec![0.0;m1.width*m1.height];
    let mut value:f32=0.0;
    for y in 0..m1.width {
        for x in 0..m1.height {
            index1=x*m1.width+y;
            index2=y*m1.height+x;

            data[index2]=m1.data[index1];
        }
    }

    Matrix{
        width:m1.height,
        height:m1.width,
        data:data.clone()
    }

}
//--------------------------------------------------------------------------------------------------------------------------
pub fn mul_matrices_old(m1:& Matrix,m2:& Matrix, m:&mut Matrix ) {
                                            
    let width:usize=m.width;
    let height:usize=m.height;
    let mut index:usize=0;

    if m1.width != m2.height  {
        panic!("mul_matrices -> Matrices are not of the same dimensions");
    }

    if m1.height != m.height || m2.width!= m.width {
        panic!("mul_matrices -> Result Matrix does not have same dimensions as the inputs");
    }

    let mut value:f32=0.0;

    let mut index1:usize=0;
    let mut index2:usize=0;

    for y in 0..height {
        for x in 0..width {
            
            value=0.0;
            //println!("y:{ } x:{ } ",y,x);
            for z in 0..m1.width {

                index1=y*m1.width+z;
                index2=z*m2.width+x;
                
               // println!("z:{ } a[{ }][{ }] i1:{ } b[{ }][{ }] i2:{ }",
                 //       z,y,z,index1,z,y,index2);

                value+=m1.data[index1]*m2.data[index2];
            }

            index=width*y+x;
            m.data[index]=value;
        }
    }
   
}
//---------------------------------------------------------------------------------------------------------------------------------------------
pub fn mul_matrices(m1:& Matrix,m2:& Matrix)-> Matrix {
                                            
    let width:usize=m2.width;
    let height:usize=m1.height;
    let mut index:usize=0;

    if m1.width != m2.height  {
        panic!("mul_matrices -> Matrices are not of the same dimensions");
    }


    let mut value:f32=0.0;

    let mut index1:usize=0;
    let mut index2:usize=0;
    let dim:usize=width*height;
    let mut data:Vec<f32>=vec![0.0;dim];

    for y in 0..height {
        for x in 0..width {
            
            value=0.0;
            //println!("y:{ } x:{ } ",y,x);
            for z in 0..m1.width {

                index1=y*m1.width+z;
                index2=z*m2.width+x;
                
               // println!("z:{ } a[{ }][{ }] i1:{ } b[{ }][{ }] i2:{ }",
                        //z,y,z,index1,z,y,index2);

                value+=m1.data[index1]*m2.data[index2];
            }

            index=width*y+x;
            data[index]=value;
        }
    }

    Matrix{
        width:width,
        height:height,
        data:data.clone()
    }
   
}
//-------------------------------------------------------------------------------------------------------------------------
pub fn sigmoid(value:f32)->f32 {

    let exp_value=-value;
    let denom=( 1.0 + exp_value.exp());

    if denom==0.0 {
        return 0.0;
    }

    let result:f32= 1.0/denom;

    result
}
//-------------------------------------------------------------------------------------------------------------------------
pub fn sigmoid_deriv(value:f32)->f32 {

    let neg_value=-value;
    let exp_value=neg_value.exp();
    let denom=1.0+exp_value;
    let denom_squared=denom.powf(2.0);
    
    if denom_squared==0.0 {
        return 0.0;
    }
    let result:f32= exp_value/( denom_squared );

    result
}
//--------------------------------------------------------------------------------------------------------------------------
pub fn as_u32_be(array: &[u8; 4]) -> u32 {
    ((array[0] as u32) << 24) +
    ((array[1] as u32) << 16) +
    ((array[2] as u32) <<  8) +
    ((array[3] as u32) <<  0)
}
//--------------------------------------------------------------------------------------------------------------------------
pub fn as_u32_le(array: &[u8; 4]) -> u32 {
    ((array[0] as u32) <<  0) +
    ((array[1] as u32) <<  8) +
    ((array[2] as u32) << 16) +
    ((array[3] as u32) << 24)
}

//--------------------------------------------------------------------------------------------------------------------------
pub fn some_func(v: f32)->f32 {
    3.0
}
//--------------------------------------------------------------------------------------------------------------------------
pub fn read_mnist_train_labels_file(filename:&str,label_data:&mut Vec<u8>)->io::Result<()>{ 
    
    let mut f = File::open(filename)?;

    let mut buffer=[0;4];
    let mut item_no:[u8;4]=[0;4];
    let mut items:u32;


    f.read(&mut buffer[..])?;
    
    f.read(&mut item_no[..])?;
   
    items=as_u32_be(&item_no);

    println!("Results");
    println!("items:{}",items);


    label_data.clear();
    label_data.resize(items as usize,0);

    f.read(&mut label_data[..])?;

    //for i in 0..9 {
      //  println!("array :{}",label_data[i]);
   // }

    Ok(())
}
//--------------------------------------------------------------------------------------------------------------------------
pub fn read_mnist_train_images_file(filename:&str,image_data:& mut Vec<ImData>)->io::Result<()>{ 
    
    let mut f = File::open(filename)?;

    let mut buffer=[0;4];
    let mut item_no:[u8;4]=[0;4];
    let mut rows_no:[u8;4]=[0;4];
    let mut cols_no:[u8;4]=[0;4];
    let mut items:u32;
    let mut rows:u32;
    let mut cols:u32;

    f.read(&mut buffer[..])?;
    f.read(&mut item_no[..])?;
    f.read(&mut rows_no[..])?;
    f.read(&mut cols_no[..])?;

    items=as_u32_be(&item_no);
    rows=as_u32_be(&rows_no);
    cols=as_u32_be(&cols_no);

    println!("Results");
    println!("items:{} cols:{} rows:{}",items,cols,rows);
    
    let total_size=rows*cols;
    //let mut image_data_:Vec<u8>=vec![0; total_size as usize]; 
    let mut image_data_:Vec<u8>=vec![0; total_size as usize]; 


    
    for i in 0..items {

        image_data.push(
        
            ImData{
                width:cols as usize,
                height:rows as usize,
                data:vec![0;total_size as usize]
            }
        );

        f.read(&mut image_data[i as usize].data[..])?;
           
    }

    Ok(())
}

//--------------------------------------------------------------------------------------------------------------------------
pub fn test_stuff(){
       
    let width:usize=2;
    let height:usize=3;

    let mut m1=Matrix{
        width:3,
        height:2,
        data:vec![0.0;6]
    };
   
    let size_=m1.data.len();
    println!("size:{} ",size_);
    
    let mut m2=Matrix{
        width:2,
        height:3,
        data:vec![0.0;6]
    };

    let mut m3=Matrix{
        width:2,
        height:2,
        data:vec![0.0;4]
    };
    
    let mut m4=Matrix{
        width:2,
        height:3,
        data:vec![0.0;6]
    };

    let mut m5=Matrix{
        width:2,
        height:2,
        data:vec![1.0;6]
    };


    let mut m6=Matrix{
        width:2,
        height:2,
        data:vec![1.0;6]
    };


   m1.data[0]=1.0;
   m1.data[1]=2.0;
   m1.data[2]=3.0;

   m1.data[3]=4.0;
   m1.data[4]=5.0;
   m1.data[5]=6.0; 

   m2.data[0]=7.0;
   m2.data[1]=8.0;
   m2.data[2]=9.0;

   m2.data[3]=10.0;
   m2.data[4]=11.0;
   m2.data[5]=12.0;
   
   println!("m1");
    debug_matrix(& m1); 

    println!("m2");
    debug_matrix(& m2);

    println!("multiply");
    m3=mul_matrices(& m1,& m2);
    debug_matrix(& m3);


    println!("debug m5");
  //  m5.debug_matrix();

    m6=add_matrices(&m5,&m3);
    println!("debug m6");
    //m6.debug_matrix();
    
    println!("transpose");
    m4=transpose_matrix(&m1);
    debug_matrix(& m4);
    
    println!("test map");

    let mut my_data=vec![1.0,2.0,3.0];

    let mut rng=rand::thread_rng();

    let mut v:Vec<f32>=my_data.iter().map( |x| rng.gen_range(0.0,1.0) ).collect();
    
    v.iter_mut().for_each( |x| *x=some_func(*x) );

    let mut new_v=vec![1.0,2.0,3.0,4.0];
    new_v=v.clone();
    new_v[0]=10.0;

    new_v.clear();
    new_v.resize(2,6.0);
    println!("{:?} ",v);
    println!("{:?}",new_v);
   // debug_matrix(& m4);    

}

pub fn add_two(x:i32)->i32 {
    x+2
}
//--------------------------------------------------------------------------------------------------------------------------
fn to_byte_slice_f32<'a>(floats: &'a [f32],size:usize) -> &'a [u8] {
    unsafe {
        std::slice::from_raw_parts(floats.as_ptr() as *const _, floats.len() * size)
    }
}
//--------------------------------------------------------------------------------------------------------------------------
 pub fn save_data(filename:&str){

     let mut f=match OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(filename){
                    Ok(file)=> file,
                    Err(..) => panic!("Error reading file:{}",filename)
        
            };
    

    let vf:Vec<f32>=vec![1.1,2.2,3.3,4.4,5.5,6.6];
    let vu=to_byte_slice_f32(&vf,4);

    println!("slice : {:?}",vu);
    
    let mut ib:[u8;4]=[0;4];
    LittleEndian::write_u32(& mut ib,100);
    f.write(&ib);

    LittleEndian::write_u32(& mut ib,200);;
    f.write(&ib);

    f.write(&vu);
    
 

 }
//-------------------------------------------------------------------------------------------------------------------------
pub fn read_data(filename:&str){
        let mut f=match OpenOptions::new()
            .read(true)
            .open(filename){
                    Ok(file)=> file,
                    Err(..) => panic!("Error reading file:{}",filename)
        
            };
     
    let mut ibuffer:[u8;4]=[0;4];
    let mut value:u32=0;

    f.read(& mut ibuffer);
    value=LittleEndian::read_u32(&ibuffer);
    println!("value1:{}",value);
    f.read(& mut ibuffer);
    value=LittleEndian::read_u32(&ibuffer);
    println!("value2:{}",value);


     let mut vu2:[u8;24]=[0;24];
     f.read(& mut vu2);
     println!("raw u8: {:?}",vu2);
    
    let mut numbers_got:[f32;6] = [0.0; 6];
    LittleEndian::read_f32_into_unchecked(&vu2, &mut numbers_got);
    println!("Data read array f32: {:?}",numbers_got);
}
//--------------------------------------------------------------------------------------------------------------------------
fn main() {
  
    
    //println!("Test");
    //save_data("nn_data.txt");
    //read_data("nn_data.txt");
    //return;

    println!("Load Training Data");

    let mut label_data:Vec<u8>=vec![0;10];
    let mut image_data:Vec<ImData>=Vec::new();
    let mut actual_labels:Vec<Vec<f32>>=Vec::new();
    
    //Init true label data
    for i in 0..10 {
        actual_labels.push(
                vec![0.0;10]
            );
        actual_labels[i]=vec![0.0;10];
        actual_labels[i][i]=1.0;
    }
    
     // 28*28=784 input neurons (images are 28*28 pixels)
    // 15 hidden neurons (experimental)
    // 10 output neurons (for each image output is a vector of size 10, full of zeros and a 1 at the index of the number represented)
    // 0.7 learning rate (experimental)
    //
    //
    let mut nn=NeuralNet::new(784,15,10,0.7);
    nn.init_net();
    nn.save_to_file("nn_data.txt");

    println!("Loading data..................");
    nn.read_from_file("nn_data.txt");

    return;
    read_mnist_train_labels_file("C:\\Users\\Costas\\Downloads\\training data\\train-labels.idx1-ubyte",& mut label_data);
    read_mnist_train_images_file("C:\\Users\\Costas\\Downloads\\training data\\train-images.idx3-ubyte",& mut image_data);   
   
    let total_size=image_data[0].width*image_data[0].height;

    let mut output_data:Vec<f32>=vec![0.0;total_size];
    
    output_data=image_data[0].data.iter_mut().map( |x|  *x as  f32 ).collect();
    let start = Instant::now();

    let mut index:usize=0;
    for i in 0..1000 {
;
        index=label_data[i] as usize;
        output_data=image_data[i].data.iter_mut().map( |x|  *x as  f32 ).collect();
        
  //      println!("item:{} index:{}==================================================================================================",i,index);

        nn.compute_output(&output_data);  
        nn.learn(&actual_labels[index]);
//        println!("output {:?}",nn.Y.data);
    }
    
    nn.save_to_file("nn_data.txt");
    

    let duration = start.elapsed();
    println!("Time elapsed in expensive_function() is: {:?}", duration );

    //println!("output {:?}",nn.Y.data)

    //println!("Image !!!");
    //image_data[2].debug_matrix();
    //println!("Data loaded!");
}
