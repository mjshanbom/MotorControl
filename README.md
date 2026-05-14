# MotorControl

## The code base

The MotorControl Project contains the code for four executable programs:

MotorControlGUI - Connects to the Motor via COM, and has simple JOG controls

ScopeGUI - The equivalent to ReadScope. Connects to RIGOL scope via VISA.

ScanGUIMulti - Currently the best functioning scan program. Allows for a lot of customizable settings

ScanGUI - Designed to be a simplified Scan, but not fully working.

When developing this program, I worked locally in C:\Users\sigma\Dropbox\QMS\TEST LABORATORY\3. OPTICS AND ENDOSCOPES\10. TOOLS\Code\MotorControl
This ensured that all of the program exes would sync to dropbox. Currently, program exes do not sync to Github as their compilation changes based on OS

This program was developed entirely in Python using claude code. To start working on modifying this software, I reccomend using VSCode. Modifications are only reccomended to those who have an understanding of Python and Terminal basics.
A requirements file is in the repository, and all required packages can be installed using pip install -r requirements.txt

After modifying code, to recompile, run PyInstaller {specfilehere}


## Using the Code

### MotorControlGUI.exe

This is the base standard JOG controller for the motor. 

<img width="771" height="738" alt="Screenshot 2026-05-14 093039" src="https://github.com/user-attachments/assets/a5a864b6-e530-4e71-9ecd-5ca02f0e7c65" />

Let's walk through this program piece by piece

Motor Port Indicates the COM port that the motor is currently connected to on the computer. Clicking find will list all active COM ports on the computer. There will usually only be a couple options, and one of them will work.

Alternatively, you can also check device manager, and look for the serial to USB connection

<img width="494" height="99" alt="Screenshot 2026-05-14 093737" src="https://github.com/user-attachments/assets/2fae0c2d-44aa-4241-a82a-c7219cd600d8" />

The next line, starting with baud rate, consists of details for the serial format for the motor controller. These do not need to be changed unless the motor controller is modified.

The next line, speed, also does not need to be modified for most use.

The next line allows you to select which lab you are in, as the motor controls differ between them.
Step size
The next section is the jog configuration. The motor assigned to each direction likely does not need to be changed unless the motor controller is reconfigured. Step size, in mm, can be stored and used for each motor using the + and - buttons. The position box is only accurate for changes in position made by the program, and does not sync to the controller. To that end, the home button is currently bugged and should not be used.

To start, press Connect, then Online. Currently the stop button is non functional.


### ScopeGUI.exe

ScopeGUI acts as a way to read the RIGOL DS1102 Z-E oscilloscope. Ensure the scope is connected to the PC over USB and on.


<img width="1708" height="1066" alt="Screenshot 2026-05-14 103338" src="https://github.com/user-attachments/assets/ecf5778e-441f-4c1a-95d9-d7b41f8a6aa9" />

Let's walk through this image. Similar to MotorControlGUI, there is a find button to identify the VISA for the scope. This VISA will be very obvious to indetify. Afterwards, click connect.

The first box is the channel box. This allows you to choose which channel to measure from. Note, the program allows for four channels, but our current scope only has two.

The next section is scope config. This allows you to configure the scope on the computer rather than on the scope. Note, this config only applies when clicking apply config, and does not apply when clicking capture. This can be useful if you want to modify the configuration directly from the scope, and only use the program to capture. The waveform can then be saved. The saved document starts with "quick information on the wave," then continues to points formatted the same as the existing labview program.

The capture button essentially runs a single on the scope. The run button is run on the scope and stop stops.

After taking a capture, the save button can be clicked to save it. Additionally, information on the wave is displayed in the window. Pii is based on the hydrophone calibration file stored in the same directory as the executable, called data.txt.



