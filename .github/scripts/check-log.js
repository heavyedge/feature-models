'use strict';

const fs = require('fs');

const MAX_CHECK_LOG_BYTES = 59_000;

function redactLog(log, secrets = []) {
  let redacted = log;
  for (const secret of [...new Set(secrets.filter((value) => value && value.length >= 3))]
    .sort((left, right) => right.length - left.length)) {
    redacted = redacted.split(secret).join('[REDACTED]');
  }

  return redacted
    .replace(/\b(?:gh[pousr]_|ghs_|github_pat_)[A-Za-z0-9_.-]+/g, '[REDACTED]')
    .replace(/\bhf_[A-Za-z0-9_-]+\b/g, '[REDACTED]')
    .replace(/(authorization:\s*(?:basic|bearer)\s+)[^\s]+/gi, '$1[REDACTED]')
    .replace(/(x-access-token:)[^\s@]+/gi, '$1[REDACTED]');
}

function truncateLog(log) {
  const bytes = Buffer.from(log, 'utf8');
  if (bytes.length <= MAX_CHECK_LOG_BYTES) {
    return log;
  }

  return `[log truncated; showing the final ${MAX_CHECK_LOG_BYTES} bytes]\n${bytes.subarray(-MAX_CHECK_LOG_BYTES).toString('utf8')}`;
}

function readCheckLog(logFile, secrets) {
  let log;
  try {
    log = fs.readFileSync(logFile, 'utf8');
  } catch (error) {
    log = `Unable to read Kubernetes Job log: ${error.message}\n`;
  }
  return redactLog(log, secrets);
}

function phaseLog(log, startMarker, endMarker) {
  const start = log.indexOf(startMarker);
  if (start === -1) {
    return log;
  }

  const end = endMarker ? log.indexOf(endMarker, start + startMarker.length) : -1;
  return log.slice(start, end === -1 ? undefined : end);
}

module.exports = { phaseLog, readCheckLog, truncateLog };
