// Name: Sadie Shudde
// Two body problem
// nvcc R_TwoBodyToNBodyCPU_Sadie_HW20_Extra.cu -o temp -lglut -lGLU -lGL
//To stop hit "control c" in the window you launched it from.

/*
 What to do:
 This is some crude code that moves two bodies around in a box, attracted by gravity and 
 repelled when they hit each other. Take this from a two-body problem to an N-body problem, where 
 NUMBER_OF_SPHERES is a #define that you can change. Also clean it up a bit so it is more user friendly.
*/

/*
 Purpose:
 To learn about Nbody code.
*/

/*
 Explain what you did to fix the code:
 1. Added the #define to get ther number of spheres made it 13 cause Taylor Swift duh
 2. Replaced the separate px1//px2 etc values with arrays the length of the number of spheres
 3. Rewrote set_initial_condtitions to place each sphere randomly. It wont let it place spheres overlapping and give random velocity and mass of 1
 4. Changed draw_picture to loop all spheres and obvisouly make them pink
 5. Changed keep_in_box to loop over all spheres
 6. Started all spheres with zero force then looped over every pair
 7. Changed move_bodies to loop over all spheres and update each velocity and position
 8. Made the X button actually work by moving the simulation out of the blocking loop in Display and into glutIdleFunc. nbody() now runs DRAW steps, draws, and returns so GLUT can process window events. This is mainly because
 it took me an embarssingly long time trying to figure out how to exit the program when I first ran it.
 9. YES RICHARD IT DOES ZOOM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
 Basically I copy and pasted 
 for(int i=0; i<NUMBER_OF_SPHERES; i++
 {
 }
 and 
 [i]
 
 put everything in a loop and replaced any 1's and 2's with [i] and [j] 
 
 I have long fake nails on and this was legit the worst thing ever! excpept for you know murders and all the other bad things
 But I have a pretty easy life so this still sucked!
*/

/*
Okay Dancing with the Stars time!!!!!
So Julia Stiles went home last night and I think it was deserved. It was her best dance but she has not been doing well recently.

Ezra's dance was not my favorite and I am a bit worried about him but he still did well

Harry and Jenna D both got perfect scores and I agree they were phenomenal!

Why do the judges hate Connor I just do not understand! He got the same score as Maura and her dance was so bad! He deserved at least one 9!

Amber and Tatyana were both good no comments.

Team Dance week! So basically the 8 stars left are split into 2 teams and they do an extra dance!
The Team Captains are the lowest scoring Stars so Connor and Tatyana.
Tatyana kegit chosse the worst dances for her teams its crazy but whatever like she just chose her friends.
Connor got all fo the best dances with Ezra, Harry, and Jenna.

Tatyanas team's dance was so bad like so bad and yet somehow they got all 9's
However, Connor's team did amazing like best dance of the season! And thankfully they got all 10's! This is the dance I included in submission! (I will also include Harry's Jazz cuase it was great!)
*/

// Include files
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Defines
#define XWindowSize 1000
#define YWindowSize 1000
#define STOP_TIME 10000.0
#define DT        0.0001
#define GRAVITY 0.1 
#define MASS 10.0  	
#define DIAMETER 1.0
#define SPHERE_PUSH_BACK_STRENGTH 50.0
#define PUSH_BACK_REDUCTION 0.1
#define DAMP 0.01
#define DRAW 100
#define LENGTH_OF_BOX 6.0
#define MAX_VELOCITY 5.0
#define NUMBER_OF_SPHERES 13 //Taylor Swifts favorite number

// Globals
const float XMax = (LENGTH_OF_BOX/2.0);
const float YMax = (LENGTH_OF_BOX/2.0);
const float ZMax = (LENGTH_OF_BOX/2.0);
const float XMin = -(LENGTH_OF_BOX/2.0);
const float YMin = -(LENGTH_OF_BOX/2.0);
const float ZMin = -(LENGTH_OF_BOX/2.0);
//Change these so it is one entry per sphere
float px[NUMBER_OF_SPHERES], py[NUMBER_OF_SPHERES], pz[NUMBER_OF_SPHERES]; 
float vx[NUMBER_OF_SPHERES], vy[NUMBER_OF_SPHERES], vz[NUMBER_OF_SPHERES];
float fx[NUMBER_OF_SPHERES], fy[NUMBER_OF_SPHERES], fz[NUMBER_OF_SPHERES];
float mass[NUMBER_OF_SPHERES];
//Brenna made me do this 
float Time = 0.0; //Simulation time
float CameraZ = 10.0; //This is how far back the camera sits smaller means zoomed in
int Paused = 0; // 1=Simulation paused

// Function prototypes
void set_initail_conditions();
void Drawwirebox();
void draw_picture();
void keep_in_box();
void get_forces();
void move_bodies(float);
void nbody();
void Display(void);
void reshape(int, int);
int main(int, char**);
//Once again Brennas fault
void keyboard(unsigned char, int, int);

void set_initail_conditions()
{ 
	time_t t;
	srand((unsigned) time(&t));
	int yeahBuddy;
	float dx, dy, dz, seperation;
	
	//AUTOMATION AND LOOP TIME i dont know why i put that i all caps 
	for(int i = 0; i< NUMBER_OF_SPHERES; i++)
	{
		yeahBuddy = 0;
		while(yeahBuddy == 0)
		{
			//Pick a random starting point 
			px[i] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			py[i] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			pz[i] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			
			//Make sure it doesn't get put onto of an existing shere
			yeahBuddy = 1;
			for(int j = 0; j < i; j++)
			{
				dx = px[j]-px[i]; 
				dy = py[j]-py[i];
				dz = pz[j]-pz[i];
				 
				seperation = sqrt(dx*dx + dy*dy + dz*dz);
				if(seperation < DIAMETER)
				{
					yeahBuddy = 0;
					break;
				}
			}
		}
		
			
		vx[i] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		vy[i] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		vz[i] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		
		mass[i] = 1.0;
	}
}

void Drawwirebox()
{		
	glColor3f (5.0,1.0,1.0);
	glBegin(GL_LINE_STRIP);
		glVertex3f(XMax,YMax,ZMax);
		glVertex3f(XMax,YMax,ZMin);	
		glVertex3f(XMax,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMax);
		glVertex3f(XMax,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		glVertex3f(XMin,YMax,ZMin);	
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMin,YMax,ZMax);	
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMax,YMin,ZMax);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMin);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMax,ZMin);
		glVertex3f(XMax,YMax,ZMin);		
	glEnd();
	
}

void draw_picture()
{
	float radius = DIAMETER/2.0;
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	Drawwirebox();

	//Draw every sphere with its own color but have to stay on theme
	for(int i = 0; i< NUMBER_OF_SPHERES; i++)
	{	
		double shade = (double)i/NUMBER_OF_SPHERES; //0.0 to just under 1 so its different for each sphere
		glColor3d(1.0, 0.2+0.6*shade, 0.5+0.4*shade);
		glPushMatrix();
		glTranslatef(px[i], py[i], pz[i]);
		glutSolidSphere(radius,20,20);
		glPopMatrix();
	}	
	
	glutSwapBuffers();
}

void keep_in_box()
{
	float halfBoxLength = (LENGTH_OF_BOX - DIAMETER)/2.0;
	for(int i = 0; i< NUMBER_OF_SPHERES; i++)
	{
		if(px[i] > halfBoxLength)
		{
			px[i] = 2.0*halfBoxLength - px[i];
			vx[i] = - vx[i];
		}
		else if(px[i] < -halfBoxLength)
		{
			px[i] = -2.0*halfBoxLength - px[i];
			vx[i] = - vx[i];
		}
		
		if(py[i] > halfBoxLength)
		{
			py[i] = 2.0*halfBoxLength - py[i];
			vy[i] = - vy[i];
		}
		else if(py[i] < -halfBoxLength)
		{
			py[i] = -2.0*halfBoxLength - py[i];
			vy[i] = - vy[i];
		}
				
		if(pz[i] > halfBoxLength)
		{
			pz[i] = 2.0*halfBoxLength - pz[i];
			vz[i] = - vz[i];
		}
		else if(pz[i] < -halfBoxLength)
		{
			pz[i] = -2.0*halfBoxLength - pz[i];
			vz[i] = - vz[i];
		}
	}
}

void get_forces()
{
	float dx,dy,dz,r,r2,dvx,dvy,dvz,forceMag,inout;
	//All spheres start with force at 0
	for(int i = 0; i< NUMBER_OF_SPHERES; i++)
	{
		fx[i] = 0.0;
		fy[i] = 0.0;
		fz[i] = 0.0;
	}
	//Look at the pairs of spheres once j replaced all of the 2's and starts after i which replaces the 1's
	for(int i = 0; i< NUMBER_OF_SPHERES; i++)
	{
		for(int j = i+1; j< NUMBER_OF_SPHERES; j++)
		{
			dx = px[j] - px[i];
			dy = py[j] - py[i];
			dz = pz[j] - pz[i];
						
			r2 = dx*dx + dy*dy + dz*dz;
			r = sqrt(r2);

			forceMag =  mass[i]*mass[j]*GRAVITY/r2;
					
			if (r < DIAMETER)
			{
				dvx = vx[j] - vx[i];
				dvy = vy[j] - vy[i];
				dvz = vz[j] - vz[i];
				inout = dx*dvx + dy*dvy + dz*dvz;
				if(inout <= 0.0)
				{
					forceMag +=  SPHERE_PUSH_BACK_STRENGTH*(r - DIAMETER);
				}
				else
				{
					forceMag +=  PUSH_BACK_REDUCTION*SPHERE_PUSH_BACK_STRENGTH*(r - DIAMETER);
				}
			}
			//Sphere i is pulled toward j and vise versa
			fx[i] += forceMag*dx/r;
			fy[i] += forceMag*dy/r;
			fz[i] += forceMag*dz/r;
			fx[j] -= forceMag*dx/r;
			fy[j] -= forceMag*dy/r;
			fz[j] -= forceMag*dz/r;
		}
	}
	
}

void move_bodies(float time)
{
	for(int i = 0; i< NUMBER_OF_SPHERES; i++)
	{
		if(time == 0.0)
		{
			vx[i] += 0.5*DT*(fx[i] - DAMP*vx[i])/mass[i];
			vy[i] += 0.5*DT*(fy[i] - DAMP*vy[i])/mass[i];
			vz[i] += 0.5*DT*(fz[i] - DAMP*vz[i])/mass[i];
		}
		else
		{
			vx[i] += DT*(fx[i] - DAMP*vx[i])/mass[i];
			vy[i] += DT*(fy[i] - DAMP*vy[i])/mass[i];
			vz[i] += DT*(fz[i] - DAMP*vz[i])/mass[i];
		}

		px[i] += DT*vx[i];
		py[i] += DT*vy[i];
		pz[i] += DT*vz[i];
	}
	
	keep_in_box();
}
//Once again I ask myself why do I do this
void nbody()
{	
	if(Paused) return; //Do nothing while paused
	
	//Do DRAW steps draw once then return so GLUT can do window events because I just had to be extra
	for(int step = 0; step < DRAW; step++)
	{
		get_forces();
		move_bodies(Time);
		Time += DT;
	}
	draw_picture(); //Redraw tge sogeres after each batch of steps

	if(Time >= STOP_TIME)
	{
		printf("\n DONE \n");
		exit(0);
	}
}

void Display(void)
{
	glLoadIdentity(); //Reset the camera 
	gluLookAt(0.0, 0.0, CameraZ, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	draw_picture();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, 50.0);

	glMatrixMode(GL_MODELVIEW);
}

//Keyboard function why do i do this to myself
void keyboard(unsigned char key, int x, int y)
{
	if(key == '+' || key == '=') CameraZ -= 0.5;
	if(key == '-' || key == '_') CameraZ += 0.5;
	if(key == ' ') Paused = !Paused;
	if(key == 'q') exit(0);
	
	if(CameraZ < 2.0) CameraZ = 2.0; //Unlike friends don't get in the box
	if(CameraZ > 40.0) CameraZ = 40.0; //Unlike Matt Damen in The Martian don't get lost in space
	
	glutPostRedisplay(); //Redraw with the new camera position
}

int main(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("N Body 3D");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	set_initail_conditions();
	printf("\n Simulating %d Spheres. Close the window or press control c to stop \n", NUMBER_OF_SPHERES);
	printf("\n Controls: + zoom in, - zoom out, spacebar pause, q quit \n");
	glutIdleFunc(nbody);
	glutKeyboardFunc(keyboard);
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;
}


