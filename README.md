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
