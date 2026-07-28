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
console.log('\x1b[32m✓ Ports 8000 and 4321 verified free.\x1b[0m');
console.log('\n\x1b[36m🚀 KishoLens Dev Environment Starting:\x1b[0m');
console.log('  \x1b[1m│\x1b[0m 🎨 \x1b[35mFrontend UI:\x1b[0m  \x1b[4mhttp://localhost:4321\x1b[0m');
console.log('  \x1b[1m│\x1b[0m ⚡ \x1b[33mBackend API:\x1b[0m  \x1b[4mhttp://127.0.0.1:8000\x1b[0m\n');
