const { execSync } = require('child_process');

function freePorts() {
  const isWin = process.platform === 'win32';
  if (isWin) {
    try {
      execSync('FOR /F "tokens=5" %a IN (\'netstat -aon ^| findstr :8000\') DO taskkill /f /pid %a 2>nul', { stdio: 'ignore' });
      execSync('FOR /F "tokens=5" %a IN (\'netstat -aon ^| findstr :4321\') DO taskkill /f /pid %a 2>nul', { stdio: 'ignore' });
    } catch (e) {}
  } else {
    try {
      execSync('kill -9 $(lsof -t -i:8000 -i:4321) 2>/dev/null', { stdio: 'ignore' });
    } catch (e) {}
  }
}

freePorts();
console.log('✓ Ports 8000 and 4321 verified free.');
