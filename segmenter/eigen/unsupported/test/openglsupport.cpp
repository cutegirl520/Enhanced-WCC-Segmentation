// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <main.h>
#include <iostream>
#include <GL/glew.h>
#include <Eigen/OpenGLSupport>
#include <GL/glut.h>
using namespace Eigen;




#define VERIFY_MATRIX(CODE,REF) { \
    glLoadIdentity(); \
    CODE; \
    Matrix<float,4,4,ColMajor> m; m.setZero(); \
    glGet(GL_MODELVIEW_MATRIX, m); \
    if(!(REF).cast<float>().isApprox(m)) { \
      std::cerr << "Expected:\n" << ((REF).cast<float>()) << "\n" << "got\n" << m << "\n\n"; \
    } \
    VERIFY_IS_APPROX((REF).cast<float>(), m); \
  }

#define VERIFY_UNIFORM(SUFFIX,NAME,TYPE) { \
    TYPE value; value.setRandom(); \
    TYPE data; \
    int loc = glGetUniformLocation(prg_id, #NAME); \
    VERIFY((loc!=-1) && "uniform not found"); \
    glUniform(loc,value); \
    EIGEN_CAT(glGetUniform,SUFFIX)(prg_id,loc,data.data()); \
    if(!value.isApprox(data)) { \
      std::cerr << "Expected:\n" << value << "\n" << "got\n" << data << "\n\n"; \
    } \
    VERIFY_IS_APPROX(value, data); \
  }
  
#define VERIFY_UNIFORMi(NAME,TYPE) { \
    TYPE value = TYPE::Random().eval().cast<float>().cast<TYPE::Scalar>(); \
    TYPE data; \
    int loc = glGetUniformLocation(prg_id, #NAME); \
    VERIFY((loc!=-1) && "uniform not found"); \
    glUniform(loc,value); \
    glGetUniformiv(prg_id,loc,(GLint*)data.data()); \
    if(!value.isApprox(data)) { \
      std::cerr << "Expected:\n" << value << "\n" << "got\n" << data << "\n\n"; \
    } \
    VERIFY_IS_APPROX(value, data); \
  }
  
void printInfoLog(GLuint objectID)
{
    int infologLength, charsWritten;
    GLchar *infoLog;
    glGetProgramiv(objectID,GL_INFO_LOG_LENGTH, &infologLength);
    if(infologLength > 0)
    {
        infoLog = new GLchar[infologLength];
        glGetProgramInfoLog(objectID, infologLength, &charsWritten, infoLog);
        if (charsWritten>0)
          std::cerr << "Shader info : \n" << infoLog << std::endl;
        delete[] infoLog;
    }
}

GLint createShader(const char* vtx, const char* frg)
{
  GLint prg_id = glCreateProgram();
  GLint vtx_id = glCreateShader(GL_VERTEX_SHADER);
  GLint frg_id = glCreateShader(GL_FRAGMENT_SHADER);
  GLint ok;
  
  glShaderSource(vtx_id, 1, &vtx, 0);
  glCompileShader(vtx_id);
  glGetShaderiv(vtx_id,GL_COMPILE_STATUS,&ok);
  if(!ok)
  {
    std::cerr << "vtx compilation failed\n";
  }
  
  glShaderSource(frg_id, 1, &frg, 0);
  glCompileShader(frg_id);
  glGetShaderiv(frg_id,GL_COMPILE_STATUS,&ok);
  if(!ok)
  {
    std::cerr << "frg compilation failed\n";
  }
  
  glAttachShader(prg_id, vtx_id);
  glAttachShader(prg_id, frg_id);
  glLinkProgram(prg_id);
  glGetProgramiv(prg_id,GL_LINK_STATUS,&ok);
  if(!ok)
  {
    std::cerr << "linking failed\n";
  }
  printInfoLog(prg_id);
  
  glUseProgram(prg_id);
  return prg_id;
}

void test_openglsupport()
{
  int argc = 0;
  glutInit(&argc, 0);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowPosition (0,0);
  glutInitWindowSize(10, 10);

  if(glutCreateWindow("Eigen") <= 0)
  {
    std::cerr << "Error: Unable to create GLUT Window.\n";
    exit(1);
  }
  
  glewExperimental = GL_TRUE;
  if(glewInit() != GLEW_OK)
  {
    std::cerr << "Warning: Failed to initialize GLEW\n";
  }

  Vector3f v3f;
  Matrix3f rot;
  glBegin(GL_POINTS);
  
  glVertex(v3f);
  glVertex(2*v3f+v3f);
  glVertex(rot*v3f);
  
  glEnd();
  
  // 4x4 matrices
  Matrix4f mf44; mf44.setRandom();
  VERIFY_MATRIX(glLoadMatrix(mf44), mf44);
  VERIFY_MATRIX(glMultMatrix(mf44), mf44);
  Matrix4d md44; md44.setRandom();
  VERIFY_MATRIX(glLoadMatrix(md44), md44);
  VERIFY_MATRIX(glMultMatrix(md44), md44);
  
  // Quaternion
  Quaterniond qd(AngleAxisd(internal::random<double>(), Vector3d::Random()));
  VERIFY_MATRIX(glRotate(qd), Projective3d(qd).matrix());
  
  Quaternionf qf(AngleAxisf(internal::random<double>(), Vector3f::Random()));
  VERIFY_MATRIX(glRotate(qf), Projective3f(qf).matrix());
  
  // 3D Transform
  Transform<float,3,AffineCompact> acf3; acf3.matrix().setRandom();
  VERIFY_MATRIX(glLoadMatrix(acf3), Projective3f(acf3).matrix());
  VERIFY_MATRIX(glMultMatrix(acf3), Projective3f(acf3).matrix());
  
  Transform<float,3,Affine> af3(acf3);
  VERIFY_MATRIX(glLoadMatrix(af3), Projective3f(af3).matrix());
  VERIFY_MATRIX(glMultMatrix(af3), Projective3f(af3).matrix());
  
  Transform<float,3,Projective> pf3; pf3.matrix().setRandom();
  VERIFY_MATRIX(glLoadMatrix(pf3), Projective3f(pf3).matrix());
  VERIFY_MATRIX(glMultMatrix(pf3), Projective3f(pf3).matrix());
  
  Transform<double,3,AffineCompact> acd3; acd3.matrix().setRandom();
  VERIFY_MATRIX(glLoadMatrix(acd3), Projective3d(acd3).matrix());
  VERIFY_MATRIX(glMultMatrix(acd3), Projective3d(acd3).matrix());
  
  Transform<double,3,Affine> ad3(acd3);
  VERIFY_MATRIX(glLoadMatrix(ad3), Projective3d(ad3).matrix());
  VERIFY_MATRIX(glMultMatrix(ad3), Projective3d(ad3).matrix());
  
  Transform<double,3,Projective> pd3; pd3.matrix().setRandom();
  VERIFY_MATRIX(glLoadMatrix(pd3), Projective3d(pd3).matrix());
  VERIFY_MATRIX(glMultMatrix(pd3), Projective3d(pd3).matrix());
  
  // translations (2D and 3D)
  {
    Vector2f vf2; vf2.setRandom(); Vector3f vf23; vf23 << vf2, 0;
    VERIFY_MATRIX(glTranslate(vf2), Projective3f(Translation3f(vf23)).matrix());
    Vector2d vd2; vd2.setRandom(); Vector3d vd23; vd23 << vd2, 0;
    VERIFY_MATRIX(glTranslate(vd2), Projective3d(Translation3d(vd23)).matrix());
    
    Vector3f vf3; vf3.setRandom();
    VERIFY_MATRIX(glTranslate(vf3), Projective3f(Translation3f(vf3)).matrix());
    Vector3d vd3; vd3.setRandom();
    VERIFY_MATRIX(glTranslate(vd3), Projective3d(Translation3d(vd3)).matrix());
    
    Translation<float,3> tf3; tf3.vector().setRandom();
    VERIFY_MATRIX(glTranslate(tf3), Projective3f(tf3).matrix());
    
    Translation<double,3> td3;  td3.vector().setRandom();
    VERIFY_MATRIX(glTranslate(td3), Projective3d(td3).matrix());
  }
  
  // scaling (2D and 3D)
  {
    Vector2f vf2; vf2.setRandom(); Vector3f vf23; vf23 << vf2, 1;
    VERIFY_MATRIX(glScale(vf2), Projective3f(Scaling(vf23)).matrix());
    Vector2d vd2; vd2.setRandom(); Vector3d vd23; vd23 << vd2, 1;
    VERIFY_MATRIX(glScale(vd2), Project