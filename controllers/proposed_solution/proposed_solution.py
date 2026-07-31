if __name__ == "__main__":
    # You should implement your solution here!
    # ---- SCORING PULSES: send while physically at victim, then reverse to clear ----
                    if self.scoring_pulse_timer > 0:
                        self.scoring_pulse_timer -= 1

                        if self.scoring_pulse_timer > 20:
                            # Phase 1 (ticks 80 → 21): stay stopped, broadcast score pulses
                            self.hardware.set_motor_speeds(0.0, 0.0)
                            if self.tick_counter % 8 == 0:
                                self.hardware.send_score_message(
                                    self.hardware.robot_id,
                                    [target[0], target[1], 1.0]
                                )
                                self.scoring_pulse_count += 1
                        else:
                            # Phase 2 (ticks 20 → 1): reverse to clear victim body
                            # so the next A* path starts from free space.
                            # ~20 ticks × 32ms × 0.22 m/s ≈ 0.14m clearance.
                            self.hardware.set_motor_speeds(-0.22, 0.0)

                        if self.scoring_pulse_timer <= 0:
                            # Done — select and drive to next victim
                            logger.info(f"[{t:.1f}s][{self.hardware.robot_id}] Scoring complete ({self.scoring_pulse_count} pulses). Cleared body. Moving on.")
                            self.hardware.set_motor_speeds(0.0, 0.0)
                            self.scoring_pulse_timer = 0
                            self.scoring_pulse_count = 0
                            self.scoring_victim = None
                            next_target = self._select_next_victim(pose)
                            if next_target:
                                self._claim_victim(next_target)
                                self.current_path = self._plan_path(pose, next_target)
                                self.path_idx = 0
                                self.stuck_replan_count = 0
                                self.recovery_count = 0
                                logger.info(f"[{t:.1f}s][{self.hardware.robot_id}] Next target: {next_target}")
                            else:
                                logger.info(f"[{t:.1f}s][{self.hardware.robot_id}] All victims found! -> STOP")
                                self.state = "STOP"
                        continue
    pass
