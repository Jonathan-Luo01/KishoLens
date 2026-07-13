const { spawn } = require('child_process');
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
  shell: true
});

child.on('close', (code) => {
  process.exit(code);
});
