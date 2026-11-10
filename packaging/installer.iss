; Inno Setup script: builds ClassWatch-Setup.exe from the PyInstaller output in dist\ClassWatch
#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif

[Setup]
AppId={{6B7E3C2A-5F41-4C1B-9E1D-3A8C2F0B7D11}
AppName=ClassWatch
AppVersion={#AppVersion}
AppPublisher=Zubair
AppPublisherURL=https://github.com/Zubair-DLCS/ClassWatch
DefaultDirName={localappdata}\Programs\ClassWatch
DefaultGroupName=ClassWatch
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=ClassWatch-Setup-{#AppVersion}
SetupIconFile=classwatch.ico
UninstallDisplayIcon={app}\ClassWatch.exe
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\dist\ClassWatch\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\ClassWatch"; Filename: "{app}\ClassWatch.exe"
Name: "{group}\Uninstall ClassWatch"; Filename: "{uninstallexe}"
Name: "{autodesktop}\ClassWatch"; Filename: "{app}\ClassWatch.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\ClassWatch.exe"; Description: "Start ClassWatch now"; Flags: nowait postinstall skipifsilent
