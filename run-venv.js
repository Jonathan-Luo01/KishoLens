const { spawn, execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const args = process.argv.slice(2);
if (args.length === 0) {
  console.error('Usage: node run-venv.js <command> [args...]');
  process.exit(1);
}

const isWin = process.platform === 'win32';
const binDir = isWin ? 'Scripts' : 'bin';
const venvBinPath = path.join('.venv', binDir);

const cmd = args[0];
let cmdPath = path.join(venvBinPath, cmd);
if (isWin && !cmdPath.endsWith('.exe') && fs.existsSync(cmdPath + '.exe')) {
  cmdPath += '.exe';
}

let spawnCmd = cmdPath;
let spawnArgs = args.slice(1);

// Fallback to uv run if local virtualenv binary doesn't exist
if (!fs.existsSync(cmdPath)) {
  spawnCmd = isWin ? 'uv.exe' : 'uv';
  spawnArgs = ['run', ...args];
}

const child = spawn(spawnCmd, spawnArgs, {
  stdio: 'inherit',
  detached: !isWin,
  env: {
    ...process.env,
    PYTHONWARNINGS: 'ignore::UserWarning:multiprocessing.resource_tracker',
    PYTHONUNBUFFERED: '1'
  }
});

let isExiting = false;

function terminate(signal = 'SIGINT') {
  if (isExiting) return;
  isExiting = true;

  if (child && child.pid) {
    if (isWin) {
      try {
        execSync(`taskkill /pid ${child.pid} /f /t`, { stdio: 'ignore' });
      } catch (e) {}
    } else {
      try {
        process.kill(-child.pid, 'SIGKILL');
      } catch (e) {
        try {
          process.kill(child.pid, 'SIGKILL');
        } catch (err) {}
      }
    }
  }

  setTimeout(() => {
    process.exit(0);
  }, 100);
}

process.on('SIGINT', () => terminate('SIGINT'));
process.on('SIGTERM', () => terminate('SIGTERM'));

child.on('close', (code) => {
  process.exit(code ?? 0);
});
